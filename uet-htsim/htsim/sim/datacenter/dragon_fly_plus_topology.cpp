// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include "dragon_fly_plus_topology.h"
#include "string.h"
#include "main.h"
#include "queue.h"
#include "dragon_fly_plus_switch.h"
#include "compositequeue.h"
#include "aeolusqueue.h"
#include "prioqueue.h"
#include "ecnprioqueue.h"
#include "queue_lossless.h"
#include "queue_lossless_input.h"
#include "queue_lossless_output.h"
#include "swift_scheduler.h"
#include "ecnqueue.h"

string ntoa(double n);
string itoa(uint64_t n);

DragonFlyPlusTopology::DragonFlyPlusTopology(
    uint32_t p, uint32_t l, uint32_t s, uint32_t h, linkspeed_bps linkspeed, mem_b queuesize,
    QueueLoggerFactory* logger_factory, EventList* ev, queue_type q, simtime_picosec latency, simtime_picosec switch_latency
) {
    _queuesize = queuesize;
    _logger_factory = logger_factory;
    _eventlist = ev;
    _linkspeed = linkspeed;
    qt = q;
    _hop_latency = latency;
    _switch_latency = switch_latency;
    _diameter = 3;
 
    _p = p;
    _l = l;
    _s = s;
    _h = h;

    if (_h == _l)
        _no_of_nodes = _p*_l*(_s*_h+1);

    cout << "DragonFlyPlus topology with " << _p << " hosts per router, " << _s + _l << " routers per group and " << ((_s + _l) * _h +1) << " groups, total nodes " << _no_of_nodes << endl;
    cout << "Queue type " << qt << endl;

    set_params();
    init_network();
}

DragonFlyPlusTopology::DragonFlyPlusTopology(
    uint32_t radix, linkspeed_bps linkspeed, mem_b queuesize, QueueLoggerFactory* logger_factory,
    EventList* ev, queue_type q, simtime_picosec latency, simtime_picosec switch_latency, topology_type type, const char *topo_dfp_sparse_file
) {
    _queuesize = queuesize;
    _logger_factory = logger_factory;
    _eventlist = ev;
    _linkspeed = linkspeed;
    qt = q;
    _hop_latency = latency;
    _switch_latency = switch_latency;
    _diameter = 3;
  
    _k = radix;
    _type = type;
    _topo_dfp_sparse_file = topo_dfp_sparse_file;

    if (_type == DFP_SPARSE_T) {
        uint32_t no_of_nodes = _k * _k * _k * _k / 16 + _k * _k / 4;

        set_params(no_of_nodes);
    } else if (_type == DFP_DENSE_T) {
        _no_of_nodes = (_k / 2) * (_k / 2) * (_k / 2 + 1);
        _p = _k / 2;
        _l = _k / 2;
        _s = _k / 2;
        _h = _k / 2;

        set_params();
    } else {
        cerr << "Topology type <<" << _type << ">> not valid. Valid ones are SPARSE and DENSE." << endl;
        abort();
    }
    
    init_network();
}

void DragonFlyPlusTopology::set_params(uint32_t no_of_nodes) {
    cout << "Set params " << no_of_nodes << endl;
    cout << "QueueSize " << _queuesize << endl;
    _no_of_nodes = 0;
    _h = 0;

    while (_no_of_nodes < no_of_nodes) {
        _h++;
        _p = _h;
        _s = _h;
        _l = _h;
        _no_of_nodes =  _p*_l*(_s*_h+1);
    }

    if (_no_of_nodes > no_of_nodes) {
        cerr << "Topology Error: can't have a DragonFlyPlus with " << no_of_nodes << " nodes" << endl;
        exit(1);
    }

    // now that we know the parameters, setup the topology.
    set_params();
}

void DragonFlyPlusTopology::set_params() {
    _no_of_groups = _type == DFP_DENSE_T ? _h + 1 : _s*_h+1;
    _no_of_switches = _no_of_groups * (_s * _l);
    _no_of_leafs = _no_of_groups * _l;
    _no_of_spines = _no_of_groups * _s;

    cout << "DragonFlyPlus topology with " << _p << " hosts per router, " << _l << " leaf routers per group and " << _s << " spine routers per group and " << _no_of_groups << " groups, total nodes " << _no_of_nodes << endl;
    cout << "Queue type " << qt << endl;

    leafs.resize(_no_of_leafs, NULL);
    spines.resize(_no_of_spines, NULL);

    pipes_host_leaf.resize(_no_of_nodes, vector<Pipe*>(_no_of_leafs));
    queues_host_leaf.resize(_no_of_nodes, vector<Queue*>(_no_of_leafs));

    pipes_leaf_host.resize(_no_of_leafs, vector<Pipe*>(_no_of_nodes));
    queues_leaf_host.resize(_no_of_leafs, vector<Queue*>(_no_of_nodes));

    pipes_leaf_spine.resize(_no_of_leafs, vector<Pipe*>(_no_of_spines));
    queues_leaf_spine.resize(_no_of_leafs, vector<Queue*>(_no_of_spines));

    pipes_spine_leaf.resize(_no_of_spines, vector<Pipe*>(_no_of_leafs));
    queues_spine_leaf.resize(_no_of_spines, vector<Queue*>(_no_of_leafs));

    pipes_spine_spine.resize(_no_of_spines, vector<Pipe*>(_no_of_spines));
    queues_spine_spine.resize(_no_of_spines, vector<Queue*>(_no_of_spines));

    if (_type == DFP_SPARSE_T && _topo_dfp_sparse_file) {
        _topo_dfp_sparse_cfg.resize(_no_of_groups, vector<uint32_t>(_no_of_groups - 1));

        std::ifstream fin(_topo_dfp_sparse_file);

        for (int i = 0; i < (int)_no_of_groups; i++) {
            for (int j = 0; j < (int)_no_of_groups - 1; j++) {
                fin >> _topo_dfp_sparse_cfg[i][j];
            }
        }
    }
}

Queue* DragonFlyPlusTopology::alloc_src_queue(QueueLogger* queueLogger){
    return new FairPriorityQueue(_linkspeed, memFromPkt(FEEDER_BUFFER), *_eventlist, queueLogger);
    //return new PriorityQueue(speedFromMbps((uint64_t)HOST_NIC), memFromPkt(FEEDER_BUFFER), *_eventlist, queueLogger);
    
}

Queue* DragonFlyPlusTopology::alloc_queue(QueueLogger* queueLogger, mem_b queuesize, link_direction dir, bool tor = false){
    return alloc_queue(queueLogger, _linkspeed, queuesize, dir, tor);
}

Queue* DragonFlyPlusTopology::alloc_queue(QueueLogger* queueLogger, linkspeed_bps speed, mem_b queuesize, link_direction dir, bool tor){
    if (qt==RANDOM)
        return new RandomQueue(speed, queuesize, *_eventlist, queueLogger, memFromPkt(RANDOM_BUFFER));
    else if (qt==COMPOSITE) {
        CompositeQueue *q = new CompositeQueue(speed, queuesize, *_eventlist, queueLogger, Switch::_trim_size, Switch::_disable_trim);
        if (_enable_ecn){
                if (!tor || dir == UPLINK || _enable_ecn_on_tor_downlink) {
                    // don't use ECN on ToR downlinks unless configured so.
                    q->set_ecn_thresholds(_ecn_low, _ecn_high);
                }
            }
        return q;
    }
        
    else if (qt==CTRL_PRIO)
        return new CtrlPrioQueue(speed, queuesize, *_eventlist, queueLogger);
    else if (qt==ECN)
        return new ECNQueue(speed, memFromPkt(queuesize), *_eventlist, queueLogger, memFromPkt(15));
    else if (qt==LOSSLESS)
        return new LosslessQueue(speed, queuesize, *_eventlist, queueLogger, NULL);
    else if (qt==LOSSLESS_INPUT)
        return new LosslessOutputQueue(speed, queuesize, *_eventlist, queueLogger);    
    else if (qt==LOSSLESS_INPUT_ECN)
        return new LosslessOutputQueue(speed, memFromPkt(10000), *_eventlist, queueLogger);
    else if (qt==COMPOSITE_ECN){
        if (tor) 
            return new CompositeQueue(speed, queuesize, *_eventlist, queueLogger, Switch::_trim_size, Switch::_disable_trim);
        else
            return new ECNQueue(speed, memFromPkt(2*SWITCH_BUFFER), *_eventlist, queueLogger, memFromPkt(15));
    }
    assert(0);
}

void DragonFlyPlusTopology::init_network(){
    QueueLogger* queueLogger;

    // host <-> leaf
    for (uint32_t j = 0; j < _no_of_leafs; j++) {
        for (uint32_t k = 0; k < _no_of_nodes; k++) {
            pipes_host_leaf[k][j] = NULL;
            queues_host_leaf[k][j] = NULL;
            pipes_leaf_host[j][k] = NULL;
            queues_leaf_host[j][k] = NULL;
        }
    }

    // spine <-> leaf; spine <-> spine
    for (uint32_t j = 0; j < _no_of_spines; j++) {
        for (uint32_t k = 0; k < _no_of_leafs; k++) {
            pipes_spine_leaf[j][k] = NULL;
            queues_spine_leaf[j][k] = NULL;
            pipes_leaf_spine[k][j] = NULL;
            queues_leaf_spine[k][j] = NULL;
        }
        for (uint32_t k = 0; k < _no_of_spines; k++) {
            pipes_spine_spine[j][k] = NULL;
            queues_spine_spine[j][k] = NULL;
        }
    }

    //
    //  Initiate switches for UEC
    //
    for (uint32_t j=0;j<_no_of_leafs;j++){
        leafs[j] = new DragonFlyPlusSwitch(*_eventlist, "Switch_Leaf_"+ntoa(j), DragonFlyPlusSwitch::LEAF, j, _switch_latency, this);
    }
    for (uint32_t j=0;j<_no_of_spines;j++){
        spines[j] = new DragonFlyPlusSwitch(*_eventlist, "Switch_Spine_"+ntoa(j), DragonFlyPlusSwitch::SPINE, j, _switch_latency, this);
    }

    tors = leafs;

    for (uint32_t j = 0; j < _no_of_leafs; j++) {
        // links from leafs to hosts
        for (uint32_t l = 0; l < _p; l++) {
            uint32_t k = j * _p + l;
            // Downlink
            if (_logger_factory) {
                queueLogger = _logger_factory->createQueueLogger();
            } else {
                queueLogger = NULL;
            }
          
            queues_leaf_host[j][k] = alloc_queue(queueLogger, _queuesize, DOWNLINK, true);
            queues_leaf_host[j][k]->setName("LEAF" + ntoa(j) + "->DST" + ntoa(k));
          
            pipes_leaf_host[j][k] = new Pipe(_hop_latency, *_eventlist);
            pipes_leaf_host[j][k]->setName("Pipe-LEAF" + ntoa(j)  + "->DST" + ntoa(k));
          
            // Uplink
            if (_logger_factory) {
                queueLogger = _logger_factory->createQueueLogger();
            } else {
                queueLogger = NULL;
            }
            queues_host_leaf[k][j] = alloc_src_queue(queueLogger);
            queues_host_leaf[k][j]->setName("SRC" + ntoa(k) + "->LEAF" +ntoa(j));
            
            queues_host_leaf[k][j]->setRemoteEndpoint(leafs[j]);

            leafs[j]->addPort(queues_leaf_host[j][k]);

            if (qt==LOSSLESS){
                ((LosslessQueue*)queues_leaf_host[j][k])->setRemoteEndpoint(queues_host_leaf[k][j]);
            }else if (qt==LOSSLESS_INPUT || qt == LOSSLESS_INPUT_ECN){
                //no virtual queue needed at server
                new LosslessInputQueue(*_eventlist,queues_host_leaf[k][j]);
            }
          
            pipes_host_leaf[k][j] = new Pipe(_hop_latency, *_eventlist);
            pipes_host_leaf[k][j]->setName("Pipe-SRC" + ntoa(k) + "->LEAF" + ntoa(j));
        }

        uint32_t groupid = j / _l;
        // links from leaf to spines
        for (uint32_t l = 0; l < _s; l++) {
            uint32_t k = groupid * _s + l;
            // Uplink
            if (_logger_factory) {
                queueLogger = _logger_factory->createQueueLogger();
            } else {
                queueLogger = NULL;
            }
          
            queues_leaf_spine[j][k] = alloc_queue(queueLogger, _queuesize, UPLINK, true);
            queues_leaf_spine[j][k]->setName("LEAF" + ntoa(j) + "->SPINE" + ntoa(k));
          
            pipes_leaf_spine[j][k] = new Pipe(_hop_latency, *_eventlist);
            pipes_leaf_spine[j][k]->setName("Pipe-LEAF" + ntoa(j)  + "->SPINE" + ntoa(k));
          
            // Downlink
            if (_logger_factory) {
                queueLogger = _logger_factory->createQueueLogger();
            } else {
                queueLogger = NULL;
            }
            queues_spine_leaf[k][j] = alloc_queue(queueLogger, _queuesize, DOWNLINK);
            queues_spine_leaf[k][j]->setName("SPINE" + ntoa(k) + "->LEAF" +ntoa(j));
            queues_leaf_spine[j][k]->setRemoteEndpoint(spines[k]);
            queues_spine_leaf[k][j]->setRemoteEndpoint(leafs[j]);

            leafs[j]->addPort(queues_leaf_spine[j][k]);
            spines[k]->addPort(queues_spine_leaf[k][j]);

            if (qt==LOSSLESS){
                ((LosslessQueue*)queues_leaf_spine[j][k])->setRemoteEndpoint(queues_spine_leaf[k][j]);
                ((LosslessQueue*)queues_spine_leaf[k][j])->setRemoteEndpoint(queues_leaf_spine[j][k]);
            }else if (qt==LOSSLESS_INPUT || qt == LOSSLESS_INPUT_ECN){            
                new LosslessInputQueue(*_eventlist, queues_leaf_spine[j][k]);
                new LosslessInputQueue(*_eventlist, queues_spine_leaf[k][j]);
            }
          
            pipes_spine_leaf[k][j] = new Pipe(_hop_latency, *_eventlist);
            pipes_spine_leaf[k][j]->setName("Pipe-SPINE" + ntoa(k) + "->LEAF" + ntoa(j));
        }
    }

    // spines to spines
    std::vector<std::pair<uint32_t, uint32_t>> jk_pairs; ///pairs of topology-wide spine ids that produce non-NULL entries in queues/pipes_spine_spine.

    if (_type == DFP_SPARSE_T && _topo_dfp_sparse_file) {
        for (int i = 0; i < (int)_no_of_groups; i++) {
            for (int j = 0; j < (int)_no_of_groups - 1; j++) {
                int x = _topo_dfp_sparse_cfg[i][j];
                int ind_x = std::find(_topo_dfp_sparse_cfg[x].begin(), _topo_dfp_sparse_cfg[x].end(), i) - _topo_dfp_sparse_cfg[x].begin();
                assert(ind_x < (int)_no_of_groups - 1);

                ///eg line 0 is 2 3 | 1 4
                ///   line 2 is 1 3 | 4 0
                /// so for (i, j) = (0, 0) => x = 2, ind_x = 3 (position on line x = 2 where we can find 0)

                if (x > i) {
                    jk_pairs.emplace_back((uint32_t)(_s * i + j / _h), (uint32_t)(_s * x + ind_x / _h));
                }
            }
        }
    } else {
        for (uint32_t j = 0; j < _no_of_spines; j++) {
            uint32_t group_id = j / _s;

            // global links
            // Spines from other groups connect to the same number spine if _type is DENSE
            if (_type == DFP_DENSE_T) {
                for (uint32_t target_group_id = 0; target_group_id < _no_of_groups; target_group_id++) {
                    uint32_t k = target_group_id * _s + j % _s;
                    
                    if (group_id < target_group_id) {
                        jk_pairs.emplace_back(j, k);
                    }
                }
            } else if (_type == DFP_SPARSE_T) { // Should also check for global link connection type
                for (uint32_t l = 0; l < _h; l++) {
                    uint32_t target_group_id = (j % _s) * _h + l;

                    if (target_group_id >= group_id) {
                        target_group_id++;
                        uint32_t k  = target_group_id * _s + group_id / _h;
                        jk_pairs.emplace_back(j, k);
                    }
                }
            }
        }
    }

    for (const auto& [j, k]: jk_pairs) {
        if (_logger_factory) {
            queueLogger = _logger_factory->createQueueLogger();
        } else {
            queueLogger = NULL;
        }
        queues_spine_spine[k][j] = alloc_queue(queueLogger, _queuesize, UPLINK);
        queues_spine_spine[k][j]->setName("SPINE" + ntoa(k) + "-G->SPINE" + ntoa(j));
    
        pipes_spine_spine[k][j] = new Pipe(_hop_latency, *_eventlist);
        pipes_spine_spine[k][j]->setName("Pipe-SPINE" + ntoa(k) + "-G->SPINE" + ntoa(j));
    
        // Uplink
        if (_logger_factory) {
            queueLogger = _logger_factory->createQueueLogger();
        } else {
            queueLogger = NULL;
        }
        queues_spine_spine[j][k] = alloc_queue(queueLogger, _queuesize, UPLINK);
        queues_spine_spine[j][k]->setName("SPINE" + ntoa(j) + "-G->SPINE" + ntoa(k));
        queues_spine_spine[k][j]->setRemoteEndpoint(spines[j]);
        queues_spine_spine[j][k]->setRemoteEndpoint(spines[k]);

        spines[j]->addPort(queues_spine_spine[j][k]);
        spines[k]->addPort(queues_spine_spine[k][j]);

        if (qt == LOSSLESS){
            ((LosslessQueue *)queues_spine_spine[j][k])->setRemoteEndpoint(queues_spine_spine[k][j]);
            ((LosslessQueue *)queues_spine_spine[k][j])->setRemoteEndpoint(queues_spine_spine[j][k]);
        } else if (qt == LOSSLESS_INPUT || qt == LOSSLESS_INPUT_ECN){            
            new LosslessInputQueue(*_eventlist, queues_spine_spine[j][k]);
            new LosslessInputQueue(*_eventlist, queues_spine_spine[k][j]);
        }
    
        pipes_spine_spine[j][k] = new Pipe(_hop_latency, *_eventlist);
        pipes_spine_spine[j][k]->setName("Pipe-SPINE" + ntoa(j) + "-G->SPINE" + ntoa(k));
    }

    // init thresholds for lossless operation
    if (qt == LOSSLESS) {
        for (uint32_t j=0;j<_no_of_leafs;j++){
            leafs[j]->configureLossless();
        }
        for (uint32_t j=0;j<_no_of_spines;j++){
            spines[j]->configureLossless();
        }
    }  
}

Route* DragonFlyPlusTopology::setup_uec_route(int host_nr) {
    Route *host_to_tor = new Route();
    host_to_tor->push_back(queues_host_leaf[host_nr][HOST_TOR(host_nr)]);
    host_to_tor->push_back(pipes_host_leaf[host_nr][HOST_TOR(host_nr)]);
    host_to_tor->push_back(queues_host_leaf[host_nr][HOST_TOR(host_nr)]->getRemoteEndpoint());

    return host_to_tor;
}

void DragonFlyPlusTopology::add_switch_loggers(Logfile& log, simtime_picosec sample_period) {
    for (uint32_t i = 0; i < _no_of_leafs; i++) {
        leafs[i]->add_logger(log, sample_period);
    }
    for (uint32_t i = 0; i < _no_of_spines; i++) {
        spines[i]->add_logger(log, sample_period);
    }
}

int DragonFlyPlusTopology::get_oversubscription_ratio(uint32_t route_strategy) {
    // Oversubscription when non-minimal paths are used
    // Assumed recommended DragonFly+ params
    if (route_strategy == Switch::ADAPTIVE_ROUTING || route_strategy == Switch::ECMP_ALL) {
        return 2;
    }

    // Oversubscription for only when minimal paths are used
    // Assumed recommended DragonFly+ params
    if (_type == DFP_DENSE_T) {
        return _k / 2;
    } else if (_type == DFP_SPARSE_T) {
        return _k * _k / 4;
    }

    // Should not get here
    return -1;
}

simtime_picosec DragonFlyPlusTopology::get_two_point_diameter_latency(int src, int dst) {
    simtime_picosec diameter_latency_end_point = 0;
    // TOR TIER link latencies assumed same as the rest
    simtime_picosec one_hop_delay = 2 * _hop_latency + _switch_latency;

    if (HOST_GROUP(src) != HOST_GROUP(dst)) {
        diameter_latency_end_point = get_diameter_latency();
    } else if (HOST_TOR(src) != HOST_TOR(dst)) {
        diameter_latency_end_point = 4 * _hop_latency + 3 * _switch_latency;
    } else {
        diameter_latency_end_point = one_hop_delay;
    }

    return diameter_latency_end_point;
}

vector<const Route*>* DragonFlyPlusTopology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse){
    vector<const Route*>* paths = new vector<const Route*>();

    route_t *routeout, *routeback;
  
    //cout << "Src is " << src << "   dest is " << dest << endl;
    if (HOST_TOR(src)==HOST_TOR(dest)){
        // forward path
        routeout = new Route();

        assert(queues_host_leaf[src][HOST_TOR(src)]);
        routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]);
        routeout->push_back(pipes_host_leaf[src][HOST_TOR(src)]);
        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
            routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]->getRemoteEndpoint());

        assert(queues_leaf_host[HOST_TOR(dest)][dest]);
        routeout->push_back(queues_leaf_host[HOST_TOR(dest)][dest]);
        routeout->push_back(pipes_leaf_host[HOST_TOR(dest)][dest]);

        // reverse path for RTS packets
        routeback = new Route();

        assert(queues_host_leaf[dest][HOST_TOR(dest)]);
        routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]);
        routeback->push_back(pipes_host_leaf[dest][HOST_TOR(dest)]);
        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
            routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]->getRemoteEndpoint());

        assert(queues_leaf_host[HOST_TOR(src)][src]);
        routeback->push_back(queues_leaf_host[HOST_TOR(src)][src]);
        routeback->push_back(pipes_leaf_host[HOST_TOR(src)][src]);

        routeout->set_reverse(routeback);
        routeback->set_reverse(routeout);

        //print_route(*routeout);
        paths->push_back(routeout);
        check_non_null(routeout);
        return paths;
    }
    else if (HOST_GROUP(src)==HOST_GROUP(dest)){
        //don't go up the hierarchy, stay in the group only.
        //there are multiple paths (1 for each spine) between the source and the destination.
        for (uint32_t i = 0; i < _s; i++) {
            routeout = new Route();

            assert(queues_host_leaf[src][HOST_TOR(src)]);
            routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]);
            routeout->push_back(pipes_host_leaf[src][HOST_TOR(src)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]->getRemoteEndpoint());
            
            assert(queues_leaf_spine[HOST_TOR(src)][HOST_GROUP(dest) * _s + i]);
            routeout->push_back(queues_leaf_spine[HOST_TOR(src)][HOST_GROUP(dest) * _s + i]);
            routeout->push_back(pipes_leaf_spine[HOST_TOR(src)][HOST_GROUP(dest) * _s + i]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_leaf_spine[HOST_TOR(src) * _s][HOST_GROUP(dest) + i]->getRemoteEndpoint());
        
            assert(queues_spine_leaf[HOST_GROUP(dest) * _s + i][HOST_TOR(dest)]);
            routeout->push_back(queues_spine_leaf[HOST_GROUP(dest) * _s + i][HOST_TOR(dest)]);
            routeout->push_back(pipes_spine_leaf[HOST_GROUP(dest) * _s + i][HOST_TOR(dest)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_spine_leaf[HOST_GROUP(dest) * _s + i][HOST_TOR(dest)]->getRemoteEndpoint());

            assert(queues_leaf_host[HOST_TOR(dest)][dest]);
            routeout->push_back(queues_leaf_host[HOST_TOR(dest)][dest]);
            routeout->push_back(pipes_leaf_host[HOST_TOR(dest)][dest]);
        
            // reverse path for RTS packets
            routeback = new Route();
        
            assert(queues_host_leaf[dest][HOST_TOR(dest)]);
            routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]);
            routeback->push_back(pipes_host_leaf[dest][HOST_TOR(dest)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]->getRemoteEndpoint());
        
            assert(queues_leaf_spine[HOST_TOR(dest)][HOST_GROUP(dest) *_s + i]);
            routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][HOST_GROUP(dest) *_s + i]);
            routeback->push_back(pipes_leaf_spine[HOST_TOR(dest)][HOST_GROUP(dest) *_s + i]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][HOST_TOR(dest) *_s + i]->getRemoteEndpoint());
        
            assert(queues_spine_leaf[HOST_GROUP(dest) *_s + i][HOST_TOR(src)]);
            routeback->push_back(queues_spine_leaf[HOST_GROUP(dest) *_s + i][HOST_TOR(src)]);
            routeback->push_back(pipes_spine_leaf[HOST_GROUP(dest) *_s + i][HOST_TOR(src)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_spine_leaf[HOST_GROUP(dest) *_s + i][HOST_TOR(src)]->getRemoteEndpoint());

            assert(queues_leaf_host[HOST_TOR(src)][src]);
            routeback->push_back(queues_leaf_host[HOST_TOR(src)][src]);
            routeback->push_back(pipes_leaf_host[HOST_TOR(src)][src]);
        
            routeout->set_reverse(routeback);
            routeback->set_reverse(routeout);
        
            //print_route(*routeout);
            paths->push_back(routeout);
            check_non_null(routeout);
        }
        return paths;
    }
    else {
        uint32_t srcgroup = HOST_GROUP(src);
        uint32_t dstgroup = HOST_GROUP(dest);

        if (_type == DFP_DENSE_T) {
            // L-G-L paths (minimal cost path)
            // For a DENSE globally linked DragonFlyPlus there are a total of _s (spines per group) L-G-L routes
            for (uint32_t j = 0; j < _s; j++) {
                routeout = new Route();

                // forward path
                // host-leaf
                assert(queues_host_leaf[src][HOST_TOR(src)]);
                routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]);
                routeout->push_back(pipes_host_leaf[src][HOST_TOR(src)]);
                //cout << "SRC " << src << " SW " << HOST_TOR(src) << " ";
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]->getRemoteEndpoint());

                uint32_t srcswitch, dstswitch;
                srcswitch = srcgroup * _s + j;
                dstswitch = dstgroup * _s + j;

                // leaf-spine
                assert(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
                routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
                routeout->push_back(pipes_leaf_spine[HOST_TOR(src)][srcswitch]);
                //cout << "SW " << srcswitch <<        " " ;
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]->getRemoteEndpoint());

                // spine-spine
                assert(queues_spine_spine[srcswitch][dstswitch]);
                routeout->push_back(queues_spine_spine[srcswitch][dstswitch]);
                routeout->push_back(pipes_spine_spine[srcswitch][dstswitch]);
                //cout << "SW " << dstswitch <<        " ";    
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeout->push_back(queues_spine_spine[srcswitch][dstswitch]->getRemoteEndpoint());

                // spine-leaf
                assert(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
                routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
                routeout->push_back(pipes_spine_leaf[dstswitch][HOST_TOR(dest)]);
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]->getRemoteEndpoint());

                // leaf-host
                // cout << "DEST " << dest <<        " " << endl;
                assert(queues_leaf_host[HOST_TOR(dest)][dest]);
                routeout->push_back(queues_leaf_host[HOST_TOR(dest)][dest]);
                routeout->push_back(pipes_leaf_host[HOST_TOR(dest)][dest]);

                // reverse path for RTS packets
                routeback = new Route();
        
                assert(queues_host_leaf[dest][HOST_TOR(dest)]);
                routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]);
                routeback->push_back(pipes_host_leaf[dest][HOST_TOR(dest)]);
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]->getRemoteEndpoint());
            
                assert(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
                routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
                routeback->push_back(pipes_leaf_spine[HOST_TOR(dest)][dstswitch]);
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]->getRemoteEndpoint());
            
                assert(queues_spine_spine[dstswitch][srcswitch]);
                routeback->push_back(queues_spine_spine[dstswitch][srcswitch]);
                routeback->push_back(pipes_spine_spine[dstswitch][srcswitch]);
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeback->push_back(queues_spine_spine[dstswitch][srcswitch]->getRemoteEndpoint());

                assert(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
                routeback->push_back(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
                routeback->push_back(pipes_spine_leaf[srcswitch][HOST_TOR(src)]);
                if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                    routeback->push_back(queues_spine_spine[srcswitch][HOST_TOR(src)]->getRemoteEndpoint());

                assert(queues_leaf_host[HOST_TOR(src)][src]);
                routeback->push_back(queues_leaf_host[HOST_TOR(src)][src]);
                routeback->push_back(pipes_leaf_host[HOST_TOR(src)][src]);
            
                routeout->set_reverse(routeback);
                routeback->set_reverse(routeout);

                paths->push_back(routeout);
                check_non_null(routeout);
            }
        } else if (_type == DFP_SPARSE_T) {
            // add lowest cost path first; the other paths requiring an intermediate group are after
            // L-G-L
            routeout = new Route();

            assert(queues_host_leaf[src][HOST_TOR(src)]);
            routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]);
            routeout->push_back(pipes_host_leaf[src][HOST_TOR(src)]);
            //cout << "SRC " << src << " SW " << HOST_TOR(src) << " ";
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]->getRemoteEndpoint());

            uint32_t srcswitch,dstswitch;
            //find srcswitch from srcgroup which has a path to dstgroup and dstswitch from dstgroup which has an incoming path from srcgroup.
            if (srcgroup < dstgroup){
                srcswitch = srcgroup * _s + (dstgroup-1)/_h;
                dstswitch = dstgroup * _s + srcgroup/_h;
            }
            else {
                srcswitch = srcgroup * _s + dstgroup/_h;
                dstswitch = dstgroup * _s + (srcgroup-1)/_h;
            }

            /* path from leaf TOR to srcspine */
            assert(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
            routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
            routeout->push_back(pipes_leaf_spine[HOST_TOR(src)][srcswitch]);
            //cout << "SW " << srcswitch <<        " " ;
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]->getRemoteEndpoint());

            /* path from source group to destination group */
            assert(queues_spine_spine[srcswitch][dstswitch]);
            routeout->push_back(queues_spine_spine[srcswitch][dstswitch]);
            routeout->push_back(pipes_spine_spine[srcswitch][dstswitch]);
            //cout << "SW " << dstswitch <<        " ";    
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_spine_spine[srcswitch][dstswitch]->getRemoteEndpoint());

            /* path from destination spine to destination leaf */
            assert(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
            routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
            routeout->push_back(pipes_spine_leaf[dstswitch][HOST_TOR(dest)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]->getRemoteEndpoint());

            //cout << "DEST " << dest <<        " " << endl;
            assert(queues_leaf_host[HOST_TOR(dest)][dest]);
            routeout->push_back(queues_leaf_host[HOST_TOR(dest)][dest]);
            routeout->push_back(pipes_leaf_host[HOST_TOR(dest)][dest]);

            // reverse path for RTS packets
            routeback = new Route();
        
            assert(queues_host_leaf[dest][HOST_TOR(dest)]);
            routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]);
            routeback->push_back(pipes_host_leaf[dest][HOST_TOR(dest)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]->getRemoteEndpoint());
        
            assert(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
            routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
            routeback->push_back(pipes_leaf_spine[HOST_TOR(dest)][dstswitch]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]->getRemoteEndpoint());
        
            assert(queues_spine_spine[dstswitch][srcswitch]);
            routeback->push_back(queues_spine_spine[dstswitch][srcswitch]);
            routeback->push_back(pipes_spine_spine[dstswitch][srcswitch]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_spine_spine[dstswitch][srcswitch]->getRemoteEndpoint());

            assert(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
            routeback->push_back(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
            routeback->push_back(pipes_spine_leaf[srcswitch][HOST_TOR(src)]);
            if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                routeback->push_back(queues_spine_spine[srcswitch][HOST_TOR(src)]->getRemoteEndpoint());

            assert(queues_leaf_host[HOST_TOR(src)][src]);
            routeback->push_back(queues_leaf_host[HOST_TOR(src)][src]);
            routeback->push_back(pipes_leaf_host[HOST_TOR(src)][src]);
        
            routeout->set_reverse(routeback);
            routeback->set_reverse(routeout);

            //print_route(*routeout);                                                                                            
            paths->push_back(routeout);
            check_non_null(routeout);

            // Intermediate routes
            /*
            //  DragonFly+ has 2 types
            //    L-G-G-L        passes 1 intermediate spine router from an intermediate group
            //  L-G-L-L-G-L      passes 2 intermediate spine routers from an intermediate group and 1 leaf router in that group
            */
            for (uint32_t p = 0;p < _no_of_groups; p++){
                if (p==srcgroup || p==dstgroup)
                    continue;

                uint32_t intergroup = p;
                //cout << "Groups " << srcgroup  << "  "  << intergroup << " " << dstgroup << endl;
                uint32_t srcswitch,dstswitch,interswitch1,interswitch2;
                //find srcswitch from srcgroup which has a path to intergroup and dstswitch from intergroup which has an incoming path from srcgroup.
                if (srcgroup<intergroup){
                    srcswitch = srcgroup * _s + (intergroup-1)/_h;
                    interswitch1 =  intergroup * _s + srcgroup/_h;
                }
                else {
                    srcswitch = srcgroup * _s + intergroup/_h;
                    interswitch1 =  intergroup * _s + (srcgroup-1)/_h;
                }
                //route from inter group to destination group.
                if (intergroup<dstgroup){
                    interswitch2 = intergroup * _s + (dstgroup-1)/_h;
                    dstswitch =  dstgroup * _s + intergroup/_h;
                }
                else {
                    interswitch2 = intergroup * _s + dstgroup/_h;
                    dstswitch =  dstgroup * _s + (intergroup-1)/_h;
                }
            
                // only 1 route
                if (interswitch1 == interswitch2) {
                    //add indirect paths via random group;
                    routeout = new Route();

                    // host-leaf
                    assert(queues_host_leaf[src][HOST_TOR(src)]);
                    routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]);
                    routeout->push_back(pipes_host_leaf[src][HOST_TOR(src)]);
                    //cout << "DPSRC " << src << " SW " << HOST_TOR(src) << " " << queues_host_leaf[src][HOST_TOR(src)]  << " ";
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]->getRemoteEndpoint());

                    // leaf-spine
                    assert(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
                    routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
                    routeout->push_back(pipes_leaf_spine[HOST_TOR(src)][srcswitch]);
                    //cout << "SW " << srcswitch <<        " " << queues_leaf_spine[HOST_TOR(src)][srcswitch] << " ";
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]->getRemoteEndpoint());
                
                    // spine-inter_spine
                    /* path from source group to inter group*/
                    assert(queues_spine_spine[srcswitch][interswitch1]);
                    routeout->push_back(queues_spine_spine[srcswitch][interswitch1]);
                    routeout->push_back(pipes_spine_spine[srcswitch][interswitch1]);
                    //cout << "SW " << interswitch1 <<        " ";    
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeout->push_back(queues_spine_spine[srcswitch][interswitch1]->getRemoteEndpoint());

                    // inter_spine-spine
                    /* path from inter group to destgroup*/
                    assert(queues_spine_spine[interswitch2][dstswitch]);
                    routeout->push_back(queues_spine_spine[interswitch2][dstswitch]);
                    routeout->push_back(pipes_spine_spine[interswitch2][dstswitch]);
                    //cout << "SW " << dstswitch <<        " ";
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeout->push_back(queues_spine_spine[interswitch2][dstswitch]->getRemoteEndpoint());

                    // spine-leaf
                    //cout << "SW " << HOST_TOR(dest) <<        " ";
                    assert(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
                    routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
                    routeout->push_back(pipes_spine_leaf[dstswitch][HOST_TOR(dest)]);
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]->getRemoteEndpoint());

                    // leaf-host
                    //cout << "DEST " << dest <<        " " << endl;
                    assert(queues_leaf_host[HOST_TOR(dest)][dest]);
                    routeout->push_back(queues_leaf_host[HOST_TOR(dest)][dest]);
                    routeout->push_back(pipes_leaf_host[HOST_TOR(dest)][dest]);

                    // reverse path for RTS packets
                    routeback = new Route();

                    assert(queues_host_leaf[dest][HOST_TOR(dest)]);
                    routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]);
                    routeback->push_back(pipes_host_leaf[dest][HOST_TOR(dest)]);
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]->getRemoteEndpoint());

                    assert(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
                    routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
                    routeback->push_back(pipes_leaf_spine[HOST_TOR(dest)][dstswitch]);
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]->getRemoteEndpoint());

                    assert(queues_spine_spine[dstswitch][interswitch2]);
                    routeback->push_back(queues_spine_spine[dstswitch][interswitch2]);
                    routeback->push_back(pipes_spine_spine[dstswitch][interswitch2]);
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeback->push_back(queues_spine_spine[dstswitch][interswitch2]->getRemoteEndpoint());

                    assert(queues_spine_spine[interswitch1][srcswitch]);
                    routeback->push_back(queues_spine_spine[interswitch1][srcswitch]);
                    routeback->push_back(pipes_spine_spine[interswitch1][srcswitch]);
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeback->push_back(queues_spine_spine[interswitch1][srcswitch]->getRemoteEndpoint());

                    assert(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
                    routeback->push_back(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
                    routeback->push_back(pipes_spine_leaf[srcswitch][HOST_TOR(src)]);
                    if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                        routeback->push_back(queues_spine_leaf[srcswitch][HOST_TOR(src)]->getRemoteEndpoint());

                    assert(queues_leaf_host[HOST_TOR(src)][src]);
                    routeback->push_back(queues_leaf_host[HOST_TOR(src)][src]);
                    routeback->push_back(pipes_leaf_host[HOST_TOR(src)][src]);

                    routeout->set_reverse(routeback);
                    routeback->set_reverse(routeout);

                    //print_route(*routeout);                                                                                            
                    paths->push_back(routeout);
                    check_non_null(routeout);
                } else {
                    // there is 1 route for each leaf intermediate leaf we could use
                    for (uint32_t i = 0; i < _l; i++) {
                        //add indirect paths via random group;
                        routeout = new Route();

                        // host-leaf
                        assert(queues_host_leaf[src][HOST_TOR(src)]);
                        routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]);
                        routeout->push_back(pipes_host_leaf[src][HOST_TOR(src)]);
                        //cout << "DPSRC " << src << " SW " << HOST_TOR(src) << " " << queues_host_leaf[src][HOST_TOR(src)]  << " ";
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_host_leaf[src][HOST_TOR(src)]->getRemoteEndpoint());

                        // leaf-spine
                        assert(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
                        routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]);
                        routeout->push_back(pipes_leaf_spine[HOST_TOR(src)][srcswitch]);
                        //cout << "SW " << srcswitch <<        " " << queues_leaf_spine[HOST_TOR(src)][srcswitch] << " ";
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_leaf_spine[HOST_TOR(src)][srcswitch]->getRemoteEndpoint());
                    
                        // spine-inter_spine
                        /* path from source group to inter group*/
                        assert(queues_spine_spine[srcswitch][interswitch1]);
                        routeout->push_back(queues_spine_spine[srcswitch][interswitch1]);
                        routeout->push_back(pipes_spine_spine[srcswitch][interswitch1]);
                        //cout << "SW " << interswitch1 <<        " ";    
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_spine_spine[srcswitch][interswitch1]->getRemoteEndpoint());

                        // inter_spine-inter_leaf
                        /* path within intermediate group */
                        assert(queues_spine_leaf[interswitch1][intergroup * _l + i]);
                        routeout->push_back(queues_spine_leaf[interswitch1][intergroup * _l + i]);
                        routeout->push_back(pipes_spine_leaf[interswitch1][intergroup * _l + i]);
                        //cout << "SW-leaf " << intergroup * _l + i <<        " ";    
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_spine_leaf[interswitch1][intergroup * _l + i]->getRemoteEndpoint());

                        // inter_leaf-inter_spine
                        /* path within intermediate group */
                        assert(queues_leaf_spine[intergroup * _l + i][interswitch2]);
                        routeout->push_back(queues_leaf_spine[intergroup * _l + i][interswitch2]);
                        routeout->push_back(pipes_leaf_spine[intergroup * _l + i][interswitch2]);
                        //cout << "SW " << interswitch2 <<        " ";    
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_leaf_spine[intergroup * _l + i][interswitch2]->getRemoteEndpoint());

                        // inter_spine-spine
                        /* path from inter group to destgroup*/
                        assert(queues_spine_spine[interswitch2][dstswitch]);
                        routeout->push_back(queues_spine_spine[interswitch2][dstswitch]);
                        routeout->push_back(pipes_spine_spine[interswitch2][dstswitch]);
                        //cout << "SW " << dstswitch <<        " ";
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_spine_spine[interswitch2][dstswitch]->getRemoteEndpoint());

                        // spine-leaf
                        //cout << "SW " << HOST_TOR(dest) <<        " ";
                        assert(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
                        routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]);
                        routeout->push_back(pipes_spine_leaf[dstswitch][HOST_TOR(dest)]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeout->push_back(queues_spine_leaf[dstswitch][HOST_TOR(dest)]->getRemoteEndpoint());

                        // leaf-host
                        //cout << "DEST " << dest <<        " " << endl;
                        assert(queues_leaf_host[HOST_TOR(dest)][dest]);
                        routeout->push_back(queues_leaf_host[HOST_TOR(dest)][dest]);
                        routeout->push_back(pipes_leaf_host[HOST_TOR(dest)][dest]);

                        // reverse path for RTS packets
                        routeback = new Route();

                        assert(queues_host_leaf[dest][HOST_TOR(dest)]);
                        routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]);
                        routeback->push_back(pipes_host_leaf[dest][HOST_TOR(dest)]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_host_leaf[dest][HOST_TOR(dest)]->getRemoteEndpoint());

                        assert(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
                        routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]);
                        routeback->push_back(pipes_leaf_spine[HOST_TOR(dest)][dstswitch]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_leaf_spine[HOST_TOR(dest)][dstswitch]->getRemoteEndpoint());

                        assert(queues_spine_spine[dstswitch][interswitch2]);
                        routeback->push_back(queues_spine_spine[dstswitch][interswitch2]);
                        routeback->push_back(pipes_spine_spine[dstswitch][interswitch2]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_spine_spine[dstswitch][interswitch2]->getRemoteEndpoint());

                        assert(queues_spine_leaf[interswitch2][intergroup * _l + i]);
                        routeback->push_back(queues_spine_leaf[interswitch2][intergroup * _l + i]);
                        routeback->push_back(pipes_spine_leaf[interswitch2][intergroup * _l + i]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_spine_leaf[interswitch2][intergroup * _l + i]->getRemoteEndpoint());

                        assert(queues_leaf_spine[intergroup * _l + i][interswitch1]);
                        routeback->push_back(queues_leaf_spine[intergroup * _l + i][interswitch1]);
                        routeback->push_back(pipes_leaf_spine[intergroup * _l + i][interswitch1]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_leaf_spine[intergroup * _l + i][interswitch1]->getRemoteEndpoint());

                        assert(queues_spine_spine[interswitch1][srcswitch]);
                        routeback->push_back(queues_spine_spine[interswitch1][srcswitch]);
                        routeback->push_back(pipes_spine_spine[interswitch1][srcswitch]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_spine_spine[interswitch1][srcswitch]->getRemoteEndpoint());

                        assert(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
                        routeback->push_back(queues_spine_leaf[srcswitch][HOST_TOR(src)]);
                        routeback->push_back(pipes_spine_leaf[srcswitch][HOST_TOR(src)]);
                        if (qt==LOSSLESS_INPUT || qt==LOSSLESS_INPUT_ECN)
                            routeback->push_back(queues_spine_leaf[srcswitch][HOST_TOR(src)]->getRemoteEndpoint());

                        assert(queues_leaf_host[HOST_TOR(src)][src]);
                        routeback->push_back(queues_leaf_host[HOST_TOR(src)][src]);
                        routeback->push_back(pipes_leaf_host[HOST_TOR(src)][src]);

                        routeout->set_reverse(routeback);
                        routeback->set_reverse(routeout);

                        //print_route(*routeout);
                        paths->push_back(routeout);
                        check_non_null(routeout);
                    }
                }            
            }
        } else {
            cerr << "Topology type not compatible <<" << _type << ">>" << endl;
            abort();
        }
        return paths;
    }
}

int64_t DragonFlyPlusTopology::find_switch(Queue* queue){
    for (uint32_t i=0;i<_no_of_nodes;i++)
        for (uint32_t j = 0;j<_no_of_leafs;j++)
            if (queues_host_leaf[i][j]==queue)
                return j;

    for (uint32_t i=0;i<_no_of_leafs;i++)
        for (uint32_t j = 0;j<_no_of_nodes;j++)
            if (queues_leaf_host[i][j]==queue)
                return i;

    for (uint32_t i=0;i<_no_of_leafs;i++)
        for (uint32_t j = 0;j<_no_of_spines;j++)
            if (queues_leaf_spine[i][j]==queue)
                return j;

    for (uint32_t i=0;i<_no_of_spines;i++)
        for (uint32_t j = 0;j<_no_of_leafs;j++)
            if (queues_spine_leaf[i][j]==queue)
                return i;

    for (uint32_t i=0;i<_no_of_spines;i++)
        for (uint32_t j = 0;j<_no_of_spines;j++)
            if (queues_spine_spine[i][j]==queue)
                return j;

    return -1;
}

int64_t DragonFlyPlusTopology::find_destination(Queue* queue){
    for (uint32_t i=0;i<_no_of_leafs;i++)
        for (uint32_t j = 0;j<_no_of_nodes;j++)
            if (queues_leaf_host[i][j]==queue)
                return j;

    return -1;
}



void DragonFlyPlusTopology::print_path(std::ofstream &paths, uint32_t src, const Route* route){
    paths << "SRC_" << src << " ";
  
    if (route->size()/2==2){
        paths << "SW_" << find_switch((Queue*)route->at(0)) << " ";
        paths << "DST_" << find_destination((Queue*)route->at(2)) << " ";
    } else if (route->size()/2==4){
        paths << "SW_" << find_switch((Queue*)route->at(0)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(2)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(4)) << " ";
        paths << "DST_" << find_destination((Queue*)route->at(6)) << " ";
    } else if (route->size()/2==5){
        paths << "SW_" << find_switch((Queue*)route->at(0)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(2)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(4)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(6)) << " ";
        paths << "DST_" << find_destination((Queue*)route->at(8)) << " ";
    } else if (route->size()/2==6){
        paths << "SW_" << find_switch((Queue*)route->at(0)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(2)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(4)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(6)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(8)) << " ";
        paths << "DST_" << find_destination((Queue*)route->at(10)) << " ";
    } else if (route->size()/2==8){
        paths << "SW_" << find_switch((Queue*)route->at(0)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(2)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(4)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(6)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(8)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(10)) << " ";
        paths << "SW_" << find_switch((Queue*)route->at(12)) << " ";
        paths << "DST_" << find_destination((Queue*)route->at(14)) << " ";
    } else {
        paths << "Wrong hop count " << ntoa(route->size()/2);
    }
  
    paths << endl;
}
