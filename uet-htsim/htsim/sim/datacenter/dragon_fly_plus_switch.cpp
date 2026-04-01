// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include <algorithm>

#include "dragon_fly_plus_switch.h"
#include "routetable.h"
#include "dragon_fly_plus_topology.h"
#include "callback_pipe.h"
#include "queue_lossless.h"
#include "queue_lossless_output.h"

DragonFlyPlusSwitch::DragonFlyPlusSwitch(EventList& eventlist, string s, switch_type t, uint32_t id,simtime_picosec delay, DragonFlyPlusTopology* dfp):
    Switch(eventlist, s), _topo_dfp_sparse_cfg(dfp->get_sparse_cfg_reference())
{
    _id = id;
    _type = t;
    _pipe = new CallbackPipe(delay,eventlist, this);
    _uproutes = NULL;
    _dfp = dfp;
    _crt_route = 0;
    _hash_salt = random();
    _last_choice = eventlist.now();
    _fib = new RouteTable();

    ///clasa apelata o data pentru fiecare switch din topologie.
}

int8_t DragonFlyPlusSwitch::compare_queuesize_dense(FibEntry* left, FibEntry* right){
    Route * r1= left->getEgressPort();
    assert(r1 && r1->size()>1);
    BaseQueue* q1 = dynamic_cast<BaseQueue*>(r1->at(0));
    Route * r2= right->getEgressPort();
    assert(r2 && r2->size()>1);
    BaseQueue* q2 = dynamic_cast<BaseQueue*>(r2->at(0));

    // Changed this so, for the case where we would use qs for topologies with non minimal paths
    // We want to use non minimal paths only if the minimal path has over 0.2 * max_q_size
    // For e.g.
    // Works best for DFP_DENSE, but for DFP_SPARSE is better to treat all paths as the same cost
    // min_path quantization + cost
    // 0 + 3 * 1 = 3
    // 1 + 3 * 1 = 4
    // 2 + 3 * 1 = 5
    // 9 + 3 * 1 = 12
    //
    // non_min_path+1 quantization + cost
    // 0 + 3 * 2 = 6
    // 1 + 3 * 2 = 7
    // 2 + 3 * 2 = 8
    // 9 + 3 * 2 = 15
    //
    // non_min_path+3 quantization + cost
    // 0 + 3 * 3 = 9
    // 1 + 3 * 3 = 10
    // 2 + 3 * 3 = 11
    // 9 + 3 * 3 = 18
    //
    if (q1->quantized_queuesize() + 3 * left->getCost() < q2->quantized_queuesize() + 3 * right->getCost())
        return 1;
    else if (q1->quantized_queuesize() + 3 * left->getCost() > q2->quantized_queuesize() + 3 * right->getCost())
        return -1;
    else 
        return 0;
}

int8_t DragonFlyPlusSwitch::compare_queuesize_sparse(FibEntry* left, FibEntry* right){
    Route * r1= left->getEgressPort();
    assert(r1 && r1->size()>1);
    BaseQueue* q1 = dynamic_cast<BaseQueue*>(r1->at(0));
    Route * r2= right->getEgressPort();
    assert(r2 && r2->size()>1);
    BaseQueue* q2 = dynamic_cast<BaseQueue*>(r2->at(0));

    if (q1->quantized_queuesize() < q2->quantized_queuesize())
        return 1;
    else if (q1->quantized_queuesize() > q2->quantized_queuesize())
        return -1;
    else 
        return 0;
}

void DragonFlyPlusSwitch::receivePacket(Packet& pkt){
    if (pkt.type()==ETH_PAUSE){
        EthPausePacket* p = (EthPausePacket*)&pkt;
        //I must be in lossless mode!
        //find the egress queue that should process this, and pass it over for processing. 
        for (size_t i = 0;i < _ports.size();i++){
            LosslessQueue* q = (LosslessQueue*)_ports.at(i);
            if (q->getRemoteEndpoint() && ((Switch*)q->getRemoteEndpoint())->getID() == p->senderID()){
                q->receivePacket(pkt);
                break;
            }
        }
        
        return;
    }

    if (_packets.find(&pkt)==_packets.end()){
        //ingress pipeline processing.

        _packets[&pkt] = true;

        const Route * nh = getNextHop(pkt,NULL);
        //set next hop which is peer switch.
        pkt.set_route(*nh);

        //emulate the switching latency between ingress and packet arriving at the egress queue.
        _pipe->receivePacket(pkt); 
    }
    else {
        _packets.erase(&pkt);
        
        //egress queue processing.
        //cout << "Switch type " << _type <<  " id " << _id << " pkt dst " << pkt.dst() << " dir " << pkt.get_direction() << endl;
        pkt.sendOn();
    }
};

void DragonFlyPlusSwitch::addHostPort(int addr, int flowid, PacketSink* transport_port){
    Route* rt = new Route();
    rt->push_back(_dfp->queues_leaf_host[_dfp->HOST_TOR(addr)][addr]);
    rt->push_back(_dfp->pipes_leaf_host[_dfp->HOST_TOR(addr)][addr]);
    rt->push_back(transport_port);
    _fib->addHostRoute(addr,rt,flowid);
}

uint32_t DragonFlyPlusSwitch::adaptive_route_p2c(vector<FibEntry*>* ecmp_set, int8_t (*cmp)(FibEntry*,FibEntry*)){
    uint32_t choice = 0, min = UINT32_MAX;
    uint32_t start, i = 0;
    static const uint16_t nr_choices = 2;
    
    do {
        start = random()%ecmp_set->size();

        Route * r= (*ecmp_set)[start]->getEgressPort();
        assert(r && r->size()>1);
        BaseQueue* q = (BaseQueue*)(r->at(0));
        assert(q);
        if (q->queuesize()<min){
            choice = start;
            min = q->queuesize();
        }
        i++;
    } while (i<nr_choices);
    return choice;
}

uint32_t DragonFlyPlusSwitch::adaptive_route(vector<FibEntry*>* ecmp_set, int8_t (*cmp)(FibEntry*,FibEntry*)){
    //cout << "adaptive_route" << endl;
    uint32_t choice = 0;

    uint32_t best_choices[256];
    uint32_t best_choices_count = 0;
  
    FibEntry* min = (*ecmp_set)[choice];
    best_choices[best_choices_count++] = choice;

    for (uint32_t i = 1; i< ecmp_set->size(); i++){
        int8_t c = cmp(min,(*ecmp_set)[i]);

        if (c < 0){
            choice = i;
            min = (*ecmp_set)[choice];
            best_choices_count = 0;
            best_choices[best_choices_count++] = choice;
        }
        else if (c==0){
            assert(best_choices_count<255);
            best_choices[best_choices_count++] = i;
        }        
    }

    assert (best_choices_count>=1);
    uint32_t choiceindex = random()%best_choices_count;
    choice = best_choices[choiceindex];
    //cout << "ECMP set choices " << ecmp_set->size() << " Choice count " << best_choices_count << " chosen entry " << choiceindex << " chosen path " << choice << " ";

    if (cmp==compare_flow_count){
        //for (uint32_t i = 0; i<best_choices_count;i++)
          //  cout << "pathcnt " << best_choices[i] << "="<< _port_flow_counts[(BaseQueue*)( (*ecmp_set)[best_choices[i]]->getEgressPort()->at(0))]<< " ";
        
        _port_flow_counts[(BaseQueue*)((*ecmp_set)[choice]->getEgressPort()->at(0))]++;
    }

    return choice;
}

uint32_t DragonFlyPlusSwitch::replace_worst_choice(vector<FibEntry*>* ecmp_set, int8_t (*cmp)(FibEntry*,FibEntry*),uint32_t my_choice){
    uint32_t best_choice = 0;
    uint32_t worst_choice = 0;

    uint32_t best_choices[256];
    uint32_t best_choices_count = 0;

    FibEntry* min = (*ecmp_set)[best_choice];
    FibEntry* max = (*ecmp_set)[worst_choice];
    best_choices[best_choices_count++] = best_choice;

    for (uint32_t i = 1; i< ecmp_set->size(); i++){
        int8_t c = cmp(min,(*ecmp_set)[i]);

        if (c < 0){
            best_choice = i;
            min = (*ecmp_set)[best_choice];
            best_choices_count = 0;
            best_choices[best_choices_count++] = best_choice;
        }
        else if (c==0){
            assert(best_choices_count<256);
            best_choices[best_choices_count++] = i;
        }        

        if (cmp(max,(*ecmp_set)[i])>0){
            worst_choice = i;
            max = (*ecmp_set)[worst_choice];
        }
    }

    //might need to play with different alternatives here, compare to worst rather than just to worst index.
    int8_t r = cmp((*ecmp_set)[my_choice],(*ecmp_set)[worst_choice]);
    assert(r>=0);

    if (r==0){
        assert (best_choices_count>=1);
        return best_choices[random()%best_choices_count];
    }
    else return my_choice;
}

void DragonFlyPlusSwitch::permute_paths(vector<FibEntry *>* uproutes) {
    int len = uproutes->size();
    for (int i = 0; i < len; i++) {
        int ix = random() % (len - i);
        FibEntry* tmppath = (*uproutes)[ix];
        (*uproutes)[ix] = (*uproutes)[len-1-i];
        (*uproutes)[len-1-i] = tmppath;
    }
}

Route* DragonFlyPlusSwitch::getNextHop(Packet& pkt, BaseQueue* ingress_port){
    vector<FibEntry*> *available_hops = _fib->getRoutes(pkt.dst());

    // cout << endl;
    // // cout << "Route: " << pkt.route() << endl;
    // cout << "Flow id: " << pkt.flow_id() << endl;
    // cout << "Path id: " << pkt.pathid() << endl;
    // // cout << "In port: " << ingress_port << endl;

    // cout << "Flow id: " << pkt.flow_id() << endl;
    // cout << "Switch id: " << _id << endl;
    // cout << "Switch type: " << (_type == LEAF ? "LEAF" : "SPINE") << endl;
    // DragonFlyPlusSwitch *last_switch = (DragonFlyPlusSwitch*) ((BaseQueue*)pkt.route()->at(0))->getSwitch();
    // if (last_switch != NULL) {
    //     cout << "Last switch id: " << last_switch->_id << endl;
    // } else {
    //     cout << "Last switch id: NONE" << endl;
    // }
    // cout << "Packet destination: " << pkt.dst() << endl;
    // cout << "Pkt min+1: " << ((pkt.flags() & AR_BIT_nmin_1) == AR_BIT_nmin_1) << endl;
    // cout << "Pkt min+3: " << ((pkt.flags() & AR_BIT_nmin_3) == AR_BIT_nmin_3) << endl;
    // cout << "Packet direction: " << (pkt.get_direction() == UP ? "UP" : (pkt.get_direction() == DOWN ? "DOWN" : "NONE")) << endl;

    // no route table entries for this destination. Add them to FIB or fail.
    if (!available_hops) {
        if (_dfp->getTopologyType() == DFP_DENSE_T) {
            //
            // Only DragonFlyPlus L-G-L routes were added in order not to have loops
            //
            if (_type == LEAF) {
                if (_dfp->HOST_TOR(pkt.dst()) == _id) {
                    // this host is directly connected
                    HostFibEntry* fe = _fib->getHostRoute(pkt.dst(),pkt.flow_id());
                    assert(fe);
                    pkt.set_direction(DOWN);
                    return fe->getEgressPort();
                } else {
                    if (_uproutes) {
                        _fib->setRoutes(pkt.dst(), _uproutes);
                    } else {
                        // route packet up to a spine switch in the group
                        // Only L-G-L routes (shortest path)
                        uint32_t group_id = _dfp->LEAF_GROUP(_id);
                        uint32_t spines_per_group = _dfp->getNSpinesGroup();

                        for (uint32_t k = 0; k < spines_per_group; k++) {
                            uint32_t spine_id = group_id * spines_per_group + k;
                            Route *r = new Route();
                            r->push_back(_dfp->queues_leaf_spine[_id][spine_id]);
                            assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                            r->push_back(_dfp->pipes_leaf_spine[_id][spine_id]);
                            r->push_back(_dfp->queues_leaf_spine[_id][spine_id]->getRemoteEndpoint());
                            _fib->addRoute(pkt.dst(), r, 1, UP);
                        }
                        _uproutes = _fib->getRoutes(pkt.dst());
                        permute_paths(_uproutes);
                    }
                }
            } else if (_type == SPINE) {
                uint32_t target_group = _dfp->HOST_GROUP(pkt.dst());
                uint32_t target_tor = _dfp->HOST_TOR(pkt.dst());
                uint32_t group_id = _dfp->SPINE_GROUP(_id);
                if (target_group == group_id) {
                    // must go down!
                    // down routes are considered to be the ones that have the destination in the same group
                    // this is done due to the set_direction function which does not allow change from DOWN direction to UP which happens in
                    //      DragonFlyPlus and some other topologies
                    Route *r = new Route();
                    r->push_back(_dfp->queues_spine_leaf[_id][target_tor]);
                    assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                    r->push_back(_dfp->pipes_spine_leaf[_id][target_tor]);
                    r->push_back(_dfp->queues_spine_leaf[_id][target_tor]->getRemoteEndpoint());

                    _fib->addRoute(pkt.dst(), r, 1, DOWN);
                } else {
                    // Determine the desination router that will be used
                    // this assumes only L-G-L (minimal paths are used)
                    uint32_t dst_spine = target_group * _dfp->getNSpinesGroup() + _id % _dfp->getNSpinesGroup();
                    Route *r = new Route();
                    r->push_back(_dfp->queues_spine_spine[_id][dst_spine]);
                    assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                    r->push_back(_dfp->pipes_spine_spine[_id][dst_spine]);
                    r->push_back(_dfp->queues_spine_spine[_id][dst_spine]->getRemoteEndpoint());

                    _fib->addRoute(pkt.dst(), r, 1, UP);

                    // Add L-G-G-L non minimal paths
                    uint32_t no_groups = _dfp->getNGroups();
                    for (uint32_t k = 0; k < no_groups; k++) {
                        if (k != group_id and k != target_group) {
                            uint32_t intermediate_spine = k * _dfp->getNSpinesGroup() + _id % _dfp->getNSpinesGroup();
                            Route *r = new Route();
                            r->push_back(_dfp->queues_spine_spine[_id][intermediate_spine]);
                            assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                            r->push_back(_dfp->pipes_spine_spine[_id][intermediate_spine]);
                            r->push_back(_dfp->queues_spine_spine[_id][intermediate_spine]->getRemoteEndpoint());

                            _fib->addRoute(pkt.dst(), r, 2, UP);
                        }
                    }

                    // Add intra group routing (L-G-L-G-L) paths
                    // Would require 2 AR_BITS (AR_BIT_nmin_1 and AR_BIT_nmin_3; set up as 3rd bit and 4th bit in pkt.flags())
                    uint32_t leafs_per_group = _dfp->getNLeafsGroup();
                    for (uint32_t k = 0; k < leafs_per_group; k++) {
                        uint32_t leaf_id = group_id * leafs_per_group + k;
                        Route *r = new Route();
                        r->push_back(_dfp->queues_spine_leaf[_id][leaf_id]);
                        assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                        r->push_back(_dfp->pipes_spine_leaf[_id][leaf_id]);
                        r->push_back(_dfp->queues_spine_leaf[_id][leaf_id]->getRemoteEndpoint());
                        _fib->addRoute(pkt.dst(), r, 3, UP);
                    }
                }
            }
            else {
                cerr << "Route lookup on switch with no proper type: " << _type << endl;
                abort();
            }
        } else if (_dfp->getTopologyType() == DFP_SPARSE_T) {
            //
            // Only DragonFlyPlus L-G-L routes were added in order not to have loops
            //
            if (_type == LEAF) {
                if (_dfp->HOST_TOR(pkt.dst()) == _id) {
                    //this host is directly connected!
                    HostFibEntry* fe = _fib->getHostRoute(pkt.dst(),pkt.flow_id());
                    assert(fe);
                    pkt.set_direction(DOWN);
                    return fe->getEgressPort();
                } else {
                    //route packet up to a spine switch in the group
                    // Only L-G-L routes (shortest path)
                    uint32_t group_id = _dfp->LEAF_GROUP(_id);
                    uint32_t target_group = _dfp->HOST_GROUP(pkt.dst());

                    uint32_t src_spine;
                    // Determine the spine router that will be used
                    if (group_id == target_group) {
                        for (uint32_t k = 0; k < _dfp->getNSpinesGroup(); k++) {
                            uint32_t spine_id = group_id * _dfp->getNSpinesGroup() + k;
                            Route *r = new Route();
                            r->push_back(_dfp->queues_leaf_spine[_id][spine_id]);
                            assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                            r->push_back(_dfp->pipes_leaf_spine[_id][spine_id]);
                            r->push_back(_dfp->queues_leaf_spine[_id][spine_id]->getRemoteEndpoint());
                            _fib->addRoute(pkt.dst(), r, 1, UP);
                        }
                    } else {
                        ///NOTE schimbat de aici in jos:

                        if (_topo_dfp_sparse_cfg.empty()) {
                            if (group_id < target_group)
                                src_spine = group_id * _dfp->getNSpinesGroup() + (target_group-1) / _dfp->getNGlobalLinks();
                            else
                                src_spine = group_id * _dfp->getNSpinesGroup() + target_group / _dfp->getNGlobalLinks();
                        } else {
                            ///ai nevoie doar de src_spine: da cu find in _topo_dfp_sparse_cfg[group_id] pentru target_group.
                            
                            ///jk_pairs: (_s * x + ind_x / _h);
                            int ind = std::find(_topo_dfp_sparse_cfg[group_id].begin(), _topo_dfp_sparse_cfg[group_id].end(), target_group) - _topo_dfp_sparse_cfg[group_id].begin();

                            src_spine = group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();
                        }

                        ///NOTE schimbat pana aici.

                        Route *r = new Route();
                        r->push_back(_dfp->queues_leaf_spine[_id][src_spine]);
                        assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                        r->push_back(_dfp->pipes_leaf_spine[_id][src_spine]);
                        r->push_back(_dfp->queues_leaf_spine[_id][src_spine]->getRemoteEndpoint());
                        _fib->addRoute(pkt.dst(), r, 1, UP);

                        // L-G-G-L and L-G-L-G-L paths
                        for (uint32_t next_group_id = 0; next_group_id < _dfp->getNGroups(); next_group_id++) {
                            if (group_id == next_group_id || target_group == next_group_id) continue;

                            uint32_t next_src_spine, next_spine_1, next_spine_2;

                            ///NOTE schimbat de aici in jos:

                            if (_topo_dfp_sparse_cfg.empty()) {
                                if (group_id < next_group_id) {
                                    next_src_spine = group_id * _dfp->getNSpinesGroup() + (next_group_id-1) / _dfp->getNGlobalLinks();
                                    next_spine_1 = next_group_id * _dfp->getNSpinesGroup() + group_id / _dfp->getNGlobalLinks();
                                }
                                else {
                                    next_src_spine = group_id * _dfp->getNSpinesGroup() + next_group_id / _dfp->getNGlobalLinks();
                                    next_spine_1 = next_group_id * _dfp->getNSpinesGroup() + (group_id-1) / _dfp->getNGlobalLinks();
                                }
                            } else {
                                int ind = std::find(_topo_dfp_sparse_cfg[group_id].begin(), _topo_dfp_sparse_cfg[group_id].end(), next_group_id) - _topo_dfp_sparse_cfg[group_id].begin();
                                next_src_spine = group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();

                                ind = std::find(_topo_dfp_sparse_cfg[next_group_id].begin(), _topo_dfp_sparse_cfg[next_group_id].end(), group_id) - _topo_dfp_sparse_cfg[next_group_id].begin();
                                next_spine_1 = next_group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();
                            }

                            // This is a non minimal route from the spine switch with minimal path (we mark it on the spine not in leaf)
                            if (src_spine == next_src_spine) continue;

                            if (_topo_dfp_sparse_cfg.empty()) {
                                if (next_group_id < target_group)
                                    next_spine_2 = next_group_id * _dfp->getNSpinesGroup() + (target_group-1) / _dfp->getNGlobalLinks();
                                else
                                    next_spine_2 = next_group_id * _dfp->getNSpinesGroup() + target_group / _dfp->getNGlobalLinks();
                            } else {
                                int ind = std::find(_topo_dfp_sparse_cfg[next_group_id].begin(), _topo_dfp_sparse_cfg[next_group_id].end(), target_group) - _topo_dfp_sparse_cfg[next_group_id].begin();
                                next_spine_2 = next_group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();
                            }

                            ///NOTE schimbat pana aici.

                            Route *r = new Route();
                            r->push_back(_dfp->queues_leaf_spine[_id][next_src_spine]);
                            assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                            r->push_back(_dfp->pipes_leaf_spine[_id][next_src_spine]);
                            r->push_back(_dfp->queues_leaf_spine[_id][next_src_spine]->getRemoteEndpoint());
                            // cout << "src: " << src_spine << " next_src: " << next_src_spine << " nxt_1: " << next_spine_1 << " nxt_2: " << next_spine_2 << endl;
                            if (next_spine_1 == next_spine_2)
                                _fib->addRoute(pkt.dst(), r, 2, UP);
                            else
                                _fib->addRoute(pkt.dst(), r, 3, UP);
                        }
                    }
                }
            } else if (_type == SPINE) {
                uint32_t target_group = _dfp->HOST_GROUP(pkt.dst());
                uint32_t target_tor = _dfp->HOST_TOR(pkt.dst());
                uint32_t group_id = _dfp->SPINE_GROUP(_id);
                if (target_group == group_id) {
                    // must go down!
                    // down routes are considered to be the ones that have the destination in the same group
                    // this is done due to the set_direction function which does not allow change from DOWN direction to UP which happens in
                    //      DragonFlyPlus and some other topologies
                    Route *r = new Route();
                    r->push_back(_dfp->queues_spine_leaf[_id][target_tor]);
                    assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                    r->push_back(_dfp->pipes_spine_leaf[_id][target_tor]);
                    r->push_back(_dfp->queues_spine_leaf[_id][target_tor]->getRemoteEndpoint());

                    _fib->addRoute(pkt.dst(), r, 1, DOWN);
                } else {
                    // Determine the desination router that will be used
                    // this assumes only L-G-L (minimal paths are used)
                    uint32_t src_spine, dst_spine;

                    ///NOTE schimbat de aici in jos:
                    if (_topo_dfp_sparse_cfg.empty()) {
                        if (group_id < target_group) {
                            src_spine = group_id * _dfp->getNSpinesGroup() + (target_group-1)/_dfp->getNGlobalLinks();
                            dst_spine = target_group * _dfp->getNSpinesGroup() + group_id/_dfp->getNGlobalLinks();
                        } else {
                            src_spine = group_id * _dfp->getNSpinesGroup() + target_group/_dfp->getNGlobalLinks();
                            dst_spine = target_group * _dfp->getNSpinesGroup() + (group_id-1)/_dfp->getNGlobalLinks();
                        }
                    } else {
                        int ind = std::find(_topo_dfp_sparse_cfg[group_id].begin(), _topo_dfp_sparse_cfg[group_id].end(), target_group) - _topo_dfp_sparse_cfg[group_id].begin();
                        src_spine = group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();

                        ind = std::find(_topo_dfp_sparse_cfg[target_group].begin(), _topo_dfp_sparse_cfg[target_group].end(), group_id) - _topo_dfp_sparse_cfg[target_group].begin();
                        dst_spine = target_group * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();
                    }
                    ///NOTE schimbat pana aici.

                    // cout << _id << " " << src_spine << endl;

                    if (_id == src_spine) {
                        Route *r = new Route();
                        r->push_back(_dfp->queues_spine_spine[_id][dst_spine]);
                        assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                        r->push_back(_dfp->pipes_spine_spine[_id][dst_spine]);
                        r->push_back(_dfp->queues_spine_spine[_id][dst_spine]->getRemoteEndpoint());

                        _fib->addRoute(pkt.dst(), r, 1, UP);
                    } else { // L-G-L-G-L paths
                        for (uint32_t k = 0; k < _dfp->getNLeafsGroup(); k++) {
                            uint32_t leaf_id = group_id * _dfp->getNLeafsGroup() + k;

                            Route *r = new Route();
                            r->push_back(_dfp->queues_spine_leaf[_id][leaf_id]);
                            assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                            r->push_back(_dfp->pipes_spine_leaf[_id][leaf_id]);
                            r->push_back(_dfp->queues_spine_leaf[_id][leaf_id]->getRemoteEndpoint());

                            _fib->addRoute(pkt.dst(), r, 4, UP); // Marked as UP route, but in reality a downlink
                        }
                    }

                    // L-G-G-L and L-G-L-G-L paths
                    for (uint32_t next_group_id = 0; next_group_id < _dfp->getNGroups(); next_group_id++) {
                        if (group_id == next_group_id || target_group == next_group_id) continue;

                        ///NOTE schimbat de aici in jos:

                        uint32_t next_spine_1, next_spine_2;

                        if (_topo_dfp_sparse_cfg.empty()) {
                            if (group_id < next_group_id) {
                                src_spine = group_id * _dfp->getNSpinesGroup() + (next_group_id-1) / _dfp->getNGlobalLinks();
                                next_spine_1 = next_group_id * _dfp->getNSpinesGroup() + group_id / _dfp->getNGlobalLinks();
                            }
                            else {
                                src_spine = group_id * _dfp->getNSpinesGroup() + next_group_id / _dfp->getNGlobalLinks();
                                next_spine_1 = next_group_id * _dfp->getNSpinesGroup() + (group_id-1) / _dfp->getNGlobalLinks();
                            }
                        } else {
                            int ind = std::find(_topo_dfp_sparse_cfg[group_id].begin(), _topo_dfp_sparse_cfg[group_id].end(), next_group_id) - _topo_dfp_sparse_cfg[group_id].begin();
                            src_spine = group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();

                            ind = std::find(_topo_dfp_sparse_cfg[next_group_id].begin(), _topo_dfp_sparse_cfg[next_group_id].end(), group_id) - _topo_dfp_sparse_cfg[next_group_id].begin();
                            next_spine_1 = next_group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks();
                        }

                        if (_id != src_spine) continue;

                        if (_topo_dfp_sparse_cfg.empty()) {
                            if (next_group_id < target_group)
                                next_spine_2 = next_group_id * _dfp->getNSpinesGroup() + (target_group-1) / _dfp->getNGlobalLinks();
                            else
                                next_spine_2 = next_group_id * _dfp->getNSpinesGroup() + target_group / _dfp->getNGlobalLinks();
                        } else {
                            int ind = std::find(_topo_dfp_sparse_cfg[next_group_id].begin(), _topo_dfp_sparse_cfg[next_group_id].end(), target_group) - _topo_dfp_sparse_cfg[next_group_id].begin();
                            next_spine_2 = next_group_id * _dfp->getNSpinesGroup() + ind / _dfp->getNGlobalLinks(); ///(copy paste din NOTE 2).
                        }

                        ///NOTE schimbat pana aici.

                        Route *r = new Route();
                        r->push_back(_dfp->queues_spine_spine[_id][next_spine_1]);
                        assert(((BaseQueue*)r->at(0))->getSwitch() == this);

                        r->push_back(_dfp->pipes_spine_spine[_id][next_spine_1]);
                        r->push_back(_dfp->queues_spine_spine[_id][next_spine_1]->getRemoteEndpoint());
                        if (next_spine_1 == next_spine_2)
                            _fib->addRoute(pkt.dst(), r, 2, UP);
                        else
                            _fib->addRoute(pkt.dst(), r, 3, UP);
                    }
                }
            }
            else {
                cerr << "Route lookup on switch with no proper type: " << _type << endl;
                abort();
            }
        } else {
            cerr << "Topology type <<" << _dfp->getTopologyType() << ">> not valid. Valid ones are SPARSE and DENSE." << endl;
            abort();
        }
    }

    available_hops = _fib->getRoutes(pkt.dst());

    // cout << "AR_FLAG: " << (pkt.flags() & AR_BIT_nmin_1) << endl;
    vector<FibEntry*> paths_to_use = vector<FibEntry*>();
    // Allow minimal paths
    for (uint32_t i = 0; i < available_hops->size(); i++)
        if (available_hops->at(i)->getCost() == 1)
            paths_to_use.push_back(available_hops->at(i));
    // cout << "no of paths: " << paths_to_use.size() << endl;

    if (_strategy == ADAPTIVE_ROUTING || _strategy == ECMP_ALL) {
        // If we used a non-minimal+1 hop, we could use from the non-minimal+3 routes or choose a minimal path
        // non-minimal+3 routes should be used only after re-routed to an intermediate group (using a non-minimal+1 path in case of DFP_DENSE)
        if (_dfp->getTopologyType() == DFP_DENSE_T) {
            if (!(pkt.flags() & AR_BIT_nmin_1)) {
                for (uint32_t i = 0; i < available_hops->size(); i++)
                    if (available_hops->at(i)->getCost() == 2)
                        paths_to_use.push_back(available_hops->at(i));
            }
        } else if (_dfp->getTopologyType() == DFP_SPARSE_T) {
            // Allow only minimal path from this point
            if ((pkt.flags() & (AR_BIT_nmin_1 | AR_BIT_nmin_3)) && paths_to_use.empty()) {
                // These are the minimal paths when routed to a non-minimal+3 intermediate route
                for (uint32_t i = 0; i < available_hops->size(); i++) {
                    if (available_hops->at(i)->getCost() == 4)
                        paths_to_use.push_back(available_hops->at(i));
                }
            } else if (!(pkt.flags() & (AR_BIT_nmin_1 | AR_BIT_nmin_3))) {
                for (uint32_t i = 0; i < available_hops->size(); i++)
                    if (available_hops->at(i)->getCost() == 2 || available_hops->at(i)->getCost() == 3)
                        paths_to_use.push_back(available_hops->at(i));
            }
        }
    }
    available_hops = &paths_to_use;
    assert(available_hops);

    //implement a form of ECMP hashing; might need to revisit based on measured performance.
    // cout << "Available Hops: " << available_hops->size() << endl;
    uint32_t ecmp_choice = 0;
    if (available_hops->size() > 1)
        switch(_strategy){
        case NIX:
            abort();
        case ECMP:
            ecmp_choice = freeBSDHash(pkt.flow_id(),pkt.pathid(),_hash_salt) % available_hops->size();
            break;
        case ECMP_ALL:
            ecmp_choice = freeBSDHash(pkt.flow_id(),pkt.pathid(),_hash_salt) % available_hops->size();
            break;
        case ADAPTIVE_ROUTING:
            if (pkt.size() < 100) {
                // don't bother adaptive routing the small packets - don't want to pollute the tables
                ecmp_choice = freeBSDHash(pkt.flow_id(),pkt.pathid(),_hash_salt) % available_hops->size();
                break;
            }
            if (_ar_sticky==DragonFlyPlusSwitch::PER_PACKET){
                ecmp_choice = adaptive_route(available_hops,fn);
            } 
            else if (_ar_sticky==DragonFlyPlusSwitch::PER_FLOWLET){     
                if (_flowlet_maps.find(pkt.flow_id())!=_flowlet_maps.end()){
                    FlowletInfo* f = _flowlet_maps[pkt.flow_id()];
                    
                    // only reroute an existing flow if its inter packet time is larger than _sticky_delta and
                    // and
                    // 50% chance happens. 
                    // and (commented out) if the switch has not taken any other placement decision that we've not seen the effects of.
                    if (eventlist().now() - f->_last > _sticky_delta && /*eventlist().now() - _last_choice > _pipe->delay() + BaseQueue::_update_period  &&*/ random()%2==0){ 
                        //cout << "AR 1 " << timeAsUs(eventlist().now()) << endl;
                        uint32_t new_route = adaptive_route(available_hops,fn); 
                        if (fn(available_hops->at(f->_egress),available_hops->at(new_route)) < 0){
                            f->_egress = new_route;
                            _last_choice = eventlist().now();
                            //cout << "Switch " << _type << ":" << _id << " choosing new path "<<  f->_egress << " for " << pkt.flow_id() << " at " << timeAsUs(eventlist().now()) << " last is " << timeAsUs(f->_last) << endl;
                        }
                    }
                    ecmp_choice = f->_egress;

                    f->_last = eventlist().now();
                }
                else {
                    //cout << "AR 2 " << timeAsUs(eventlist().now()) << endl;
                    ecmp_choice = adaptive_route(available_hops,fn); 
                    _last_choice = eventlist().now();

                    _flowlet_maps[pkt.flow_id()] = new FlowletInfo(ecmp_choice,eventlist().now());
                }
            }

            break;
        case ECMP_ADAPTIVE:
            ecmp_choice = freeBSDHash(pkt.flow_id(),pkt.pathid(),_hash_salt) % available_hops->size();
            if (random()%100 < 50)
                ecmp_choice = replace_worst_choice(available_hops,fn, ecmp_choice);
            break;
        case RR:
            if (pkt.size()<128)
                ecmp_choice = freeBSDHash(pkt.flow_id(),pkt.pathid(),_hash_salt) % available_hops->size();
            else {
                if (_crt_route>=1*available_hops->size()){
                    _crt_route = 0;
                    permute_paths(available_hops);
                }
                ecmp_choice = _crt_route % available_hops->size();
                _crt_route ++;
            }
            break;
        case RR_ECMP:
            if (_type == LEAF){
                if (_crt_route>=5 * available_hops->size()){
                    _crt_route = 0;
                    permute_paths(available_hops);
                }
                ecmp_choice = _crt_route % available_hops->size();
                _crt_route ++;
            }
            else ecmp_choice = freeBSDHash(pkt.flow_id(),pkt.pathid(),_hash_salt) % available_hops->size();
            
            break;
        }
    
    // cout << "ECMP: " << ecmp_choice << endl;
    // cout << "no hops: " << available_hops->size() << endl;
    FibEntry* e = (*available_hops)[ecmp_choice];
    // cout << "next hop direction: " << (e->getDirection() == UP ? "UP" : (e->getDirection() == DOWN ? "DOWN" : "NONE")) << endl;
    // cout << "egress port: " << e->getEgressPort() << endl;
    pkt.set_direction(e->getDirection());

    // Set adaptive route bit to ON
    if (e->getCost() > 1 && _type == SPINE) {
        if (_dfp->getTopologyType() == DFP_DENSE_T) {
            pkt.set_flags(pkt.flags() | AR_BIT_nmin_1);
            if (e->getCost() == 3)
                pkt.set_flags(pkt.flags() | AR_BIT_nmin_3);
        } else if (_dfp->getTopologyType() == DFP_SPARSE_T) {
            if (e->getCost() == 2)
                pkt.set_flags(pkt.flags() | AR_BIT_nmin_1);
            if (e->getCost() == 3)
                pkt.set_flags(pkt.flags() | AR_BIT_nmin_3);
        }
    }

    // cout << "available_hops ended" << endl;
    // cout << "ecmp choice: " << ecmp_choice << endl;
    return e->getEgressPort();
};
