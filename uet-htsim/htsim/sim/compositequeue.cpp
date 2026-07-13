// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        
#include "compositequeue.h"
#include <math.h>
#include <iostream>
#include <sstream>
#include "ecn.h"
#include "queue_lossless_input.h"

static int global_queue_id=0;
bool CompositeQueue::isLossless = false;
#define DEBUG_QUEUE_ID -1 // set to queue ID to enable debugging

CompositeQueue::CompositeQueue(linkspeed_bps bitrate, mem_b maxsize, EventList& eventlist, 
                               QueueLogger* logger, uint16_t trim_size, bool disable_trim)
    : Queue(bitrate, maxsize, eventlist, logger)
{
    _disable_trim = disable_trim;
    _trim_size = trim_size;
    _ratios[QUEUE_LOW_VC_0] = 4;
    _ratios[QUEUE_LOW_VC_1] = 1;
    _ratios[QUEUE_HIGH] = 100000;
    _num_headers = 0;
    _num_packets = 0;
    _num_acks = 0;
    _num_nacks = 0;
    _num_pulls = 0;
    _num_drops = 0;
    _num_stripped = 0;
    _num_bounced = 0;
    _ecn_minthresh = maxsize*2; // don't set ECN by default
    _ecn_maxthresh = maxsize*2; // don't set ECN by default

    _return_to_sender = false;

    _queuesize_high_watermark = 0;
    _serv = QUEUE_INVALID_VC;
    stringstream ss;
    ss << "compqueue(" << bitrate/1000000 << "Mb/s," << maxsize << "bytes)";
    _nodename = ss.str();
    _queue_id = global_queue_id++;
    if (_queue_id == DEBUG_QUEUE_ID)
        cout << "queueid " << _queue_id << " bitrate " << bitrate/1000000 << "Mb/s," << endl;
}

void CompositeQueue::beginService() {
    _serv = QUEUE_INVALID_VC;
    // In decreasing order as the max rank is the last queue
    for (int i = VCs - 1; i >= 0; i--) {
        if (!_queues[i].empty() && _crts[i] < _ratios[i] && (!isLossless || _states[i] == READY) ) {
            _serv = i;
            eventlist().sourceIsPendingRel(*this, drainTime(_queues[i].back()));
            _crts[i]++;
            return;
        }
    }

    // If we get here, it means that we did not choose a queue as all non empty queues have reached their weigths; reset all counters
    for (int i = 0; i < VCs; i++)
        _crts[i] = 0;

    for (int i = VCs - 1; i >= 0; i--) {
        if (!_queues[i].empty() && _crts[i] < _ratios[i] && (!isLossless || _states[i] == READY) ) {
            _serv = i;
            eventlist().sourceIsPendingRel(*this, drainTime(_queues[i].back()));
            _crts[i]++;
            break;
        }
    }
    //assert(_serv >= 0);
}

bool CompositeQueue::decide_ECN() {
    //ECN mark on deque
    assert(_serv >= 0 && _serv < VCs);
    if (_queuesize[_serv] > _ecn_maxthresh) {
        return true;
    } else if (_queuesize[_serv] > _ecn_minthresh) {
        uint64_t p = (0x7FFFFFFF * (_queuesize[_serv] - _ecn_minthresh))/(_ecn_maxthresh - _ecn_minthresh);
        if ((uint64_t)random() < p) {
            return true;
        }
    }
    return false;
}

void CompositeQueue::processPause(EthPausePacket* p) {
    assert(isLossless);

    int pkt_vc = p->vc();
    if (p->sleepTime()>0) {
        //remote end is telling us to shut up.
        //assert(_state_send == READY);
        if (_serv == pkt_vc) {
            // We have a pkt in flight, wait for it to finish sending then stop
            _states[pkt_vc] = PAUSE_RECEIVED;
        } else {
            _states[pkt_vc] = PAUSED;
        }
    }
    else {
        //we are allowed to send!
        _states[pkt_vc] = READY;

        if(_serv == QUEUE_INVALID_VC)
            beginService();
    }
}

void CompositeQueue::completeService(){
    Packet* pkt;
    if (_serv < QUEUE_HIGH) {
        assert(!_queues[_serv].empty());
        pkt = _queues[_serv].pop();
        _queuesize[_serv] -= pkt->size();

        bool ecn = decide_ECN();
        //ECN mark on deque
        if (ecn) {
            pkt->set_flags(pkt->flags() | ECN_CE);
        }
        if (_queue_id == DEBUG_QUEUE_ID) {
            cout << timeAsUs(eventlist().now()) <<" name " <<_nodename << " VC " << _serv <<" _queuesize " 
                << _queuesize[_serv]*8/((_bitrate/1000000.0)) <<" _queueid " << _queue_id << " switch " << _switch->getID() 
                << " ecn " << ecn
                << endl;    

        }
        if (_logger) _logger->logQueue(*this, QueueLogger::PKT_SERVICE, *pkt);
        _num_packets++;
    } else if (_serv==QUEUE_HIGH) {
        assert(!_queues[_serv].empty());
        pkt = _queues[_serv].pop();
        if (_queuesize[_serv] > _queuesize_high_watermark) {
            _queuesize_high_watermark = _queuesize[_serv];
        }
        _queuesize[_serv] -= pkt->size();
        if (_logger) _logger->logQueue(*this, QueueLogger::PKT_SERVICE, *pkt);
        if (pkt->type() == NDPACK)
            _num_acks++;
        else if (pkt->type() == NDPNACK)
            _num_nacks++;
        else if (pkt->type() == NDPPULL)
            _num_pulls++;
        else {
            //cout << "Hdr: type=" << pkt->type() << endl;
            _num_headers++;
            //ECN mark on deque of a header, if low priority queue is still over threshold
//            if (decide_ECN()) {
//                pkt->set_flags(pkt->flags() | ECN_CE);
//            }
        }
    } else {
        assert(0);
    }

    log_packet_send(drainTime(pkt), pkt->vc());

    if (isLossless) {
        LosslessInputQueue* prev_Q = pkt->get_ingress_queue();
        prev_Q->completedService(*pkt);

        if (pkt->header_only() && _states[pkt->vc()] == PAUSE_RECEIVED) {
            _states[pkt->vc()] = PAUSED;
        } else if (!pkt->header_only() && _states[pkt->vc()] == PAUSE_RECEIVED) {
            _states[pkt->vc()] = PAUSED;
        }
    }
    pkt->clear_ingress_queue();
    
    pkt->flow().logTraffic(*pkt,*this,TrafficLogger::PKT_DEPART);
    pkt->sendOn();

    //_virtual_time += drainTime(pkt);
  
    _serv = QUEUE_INVALID_VC;
  
    if (!_queues[QUEUE_LOW_VC_0].empty() || !_queues[QUEUE_LOW_VC_1].empty() || !_queues[QUEUE_HIGH].empty())
        beginService();
}

void CompositeQueue::doNextEvent() {
    completeService();
}

void CompositeQueue::receivePacket(Packet& pkt)
{
    if (_queue_id == DEBUG_QUEUE_ID)
    {
        cout << timeAsUs(eventlist().now()) << " name " << _nodename << " arrive for VC " << pkt.vc() << " "
             << _queuesize[pkt.vc()] * 8 / ((_bitrate / 1000000.0)) << " _queueid " << _queue_id << " switch " << _switch->getID() 
             <<" flowid " << pkt.flow_id() << " ev " << pkt.pathid()<< endl;
    }

    if (pkt.type()==ETH_PAUSE) {
        EthPausePacket* p = (EthPausePacket*)&pkt;
        processPause(p);
        pkt.free();
        return;
    }
    pkt.flow().logTraffic(pkt,*this,TrafficLogger::PKT_ARRIVE);
    if (_logger) _logger->logQueue(*this, QueueLogger::PKT_ARRIVE, pkt);

    int pkt_vc = pkt.vc();
    if (pkt_vc < QUEUE_HIGH) {
        if (_queuesize[pkt_vc] + pkt.size() <= _maxsize || drand() < 0.5) {
            //regular packet; don't drop the arriving packet

            // we are here because either the queue isn't full or,
            // it might be full and we randomly chose an
            // enqueued packet to trim
            
            if (_queuesize[pkt_vc] + pkt.size() > _maxsize) {
                // we're going to drop an existing packet from the queue
                if (_queues[pkt_vc].empty()){
                    //cout << "QUeuesize " << _queuesize_low << " packetsize " << pkt.size() << " maxsize " << _maxsize << endl;
                    assert(0);
                }
                //take last packet from low prio queue, make it a header and place it in the high prio queue
                Packet* booted_pkt = _queues[pkt_vc].pop_front();
                _queuesize[pkt_vc] -= booted_pkt->size();
                if (_logger) _logger->logQueue(*this, QueueLogger::PKT_UNQUEUE, *booted_pkt);

                if (_disable_trim || isLossless) {
                    booted_pkt->free();
                    _num_drops++;
                    cout << timeAsUs(eventlist().now())
                         << " " << _nodename << " VC " << pkt_vc
                         << " A [ " << _queues[pkt_vc].size() << " ] DROP "
                         << " flowid " << booted_pkt->flow_id()<< endl;
                    assert(0);
                } else {
                    // cout << "A [ " << _enqueued_low.size() << " " << _enqueued_high.size() << " ] STRIP" << endl;
                    // cout << "booted_pkt->size(): " << booted_pkt->size();
                    booted_pkt->strip_payload(_trim_size);
                    // cout << "CQ trim at " << _nodename << endl;
                    _num_stripped++;
                    booted_pkt->flow().logTraffic(*booted_pkt, *this, TrafficLogger::PKT_TRIM);
                    if (_logger)
                        _logger->logQueue(*this, QueueLogger::PKT_TRIM, pkt);

                    if (_queuesize[QUEUE_HIGH] + booted_pkt->size() > 2 * _maxsize) {
                        if (_return_to_sender && booted_pkt->reverse_route() && booted_pkt->bounced() == false) {
                            // return the packet to the sender
                            if (_logger)
                                _logger->logQueue(*this, QueueLogger::PKT_BOUNCE, *booted_pkt);
                            booted_pkt->flow().logTraffic(pkt, *this, TrafficLogger::PKT_BOUNCE);
                            // XXX what to do with it now?
#if 0
                            printf("Bounce2 at %s\n", _nodename.c_str());
                            printf("Fwd route:\n");
                            print_route(*(booted_pkt->route()));
                            printf("nexthop: %d\n", booted_pkt->nexthop());
#endif
                            booted_pkt->bounce();
#if 0
                            printf("\nRev route:\n");
                            print_route(*(booted_pkt->reverse_route()));
                            printf("nexthop: %d\n", booted_pkt->nexthop());
#endif
                            _num_bounced++;
                            booted_pkt->sendOn();
                        } else {
                            booted_pkt->flow().logTraffic(*booted_pkt, *this, TrafficLogger::PKT_DROP);
                            booted_pkt->free();
                            if (_logger)
                                _logger->logQueue(*this, QueueLogger::PKT_DROP, pkt);
                            assert(0);
                        }
                    } else {
                        _queues[QUEUE_HIGH].push(booted_pkt);
                        _queuesize[QUEUE_HIGH] += booted_pkt->size();
                        if (_logger)
                            _logger->logQueue(*this, QueueLogger::PKT_ENQUEUE, *booted_pkt);
                    }
                }
            }

            //assert(_queuesize_low+pkt.size()<= _maxsize);
            Packet* pkt_p = &pkt;
            _queues[pkt_vc].push(pkt_p);
            _queuesize[pkt_vc] += pkt.size();
            if (_logger) _logger->logQueue(*this, QueueLogger::PKT_ENQUEUE, pkt);
            
            if (_serv==QUEUE_INVALID_VC) {
                beginService();
            }
            
            //cout << "BL[ " << _enqueued_low.size() << " " << _enqueued_high.size() << " ]" << endl;
            
            return;
        } else {
            if (_disable_trim || isLossless) {
                if (_queue_id == DEBUG_QUEUE_ID) {
                    cout <<timeAsUs(eventlist().now()) << "B[ " << _queues[pkt_vc].size() << " VC " << pkt_vc
                         << " ] DROP " << pkt.flow().flow_id() << " queue "
                         << str() << " pathid " <<pkt.pathid()<< " queueid " << _queue_id
                         << " size " << pkt.size() << endl;
                }
                cout << timeAsUs(eventlist().now())
                     << " " << _nodename
                     << " VC: " << pkt_vc
                     << " qsize_low: " << _queuesize[pkt_vc]
                     << " max_q: " << _maxsize
                     << endl;
                pkt.free();
                _num_drops++;
                assert(0);
                return;
            }
            //strip packet the arriving packet - low priority queue is full
            //cout << "B [ " << _enqueued_low.size() << " " << _enqueued_high.size() << " ] STRIP" << endl;
            pkt.strip_payload(_trim_size);
            //cout << "CQ trim at " << _nodename << endl;
            _num_stripped++;
            pkt.flow().logTraffic(pkt,*this,TrafficLogger::PKT_TRIM);
            if (_logger) _logger->logQueue(*this, QueueLogger::PKT_TRIM, pkt);
        }
    }
    assert(pkt.header_only());
    
    if (_queuesize[QUEUE_HIGH] + pkt.size() > 2*_maxsize) {
        //drop header
        //cout << "drop!\n";
        if (_return_to_sender && pkt.reverse_route()  && pkt.bounced() == false) {
            //return the packet to the sender
            if (_logger) _logger->logQueue(*this, QueueLogger::PKT_BOUNCE, pkt);
            pkt.flow().logTraffic(pkt,*this,TrafficLogger::PKT_BOUNCE);
            //XXX what to do with it now?
#if 0
            printf("Bounce1 at %s\n", _nodename.c_str());
            printf("Fwd route:\n");
            print_route(*(pkt.route()));
            printf("nexthop: %d\n", pkt.nexthop());
#endif
            pkt.bounce();
#if 0
            printf("\nRev route:\n");
            print_route(*(pkt.reverse_route()));
            printf("nexthop: %d\n", pkt.nexthop());
#endif
            _num_bounced++;
            pkt.sendOn();
            return;
        } else {
            if (_logger) _logger->logQueue(*this, QueueLogger::PKT_DROP, pkt);
            pkt.flow().logTraffic(pkt,*this,TrafficLogger::PKT_DROP);
            cout << "B[ " << _queues[QUEUE_HIGH].size() << " ] DROP " 
                 << pkt.flow().flow_id() << endl;
            pkt.free();
            _num_drops++;
            assert(0);
            return;
        }
    }
    
    
    //if (pkt.type()==NDP)
    //  cout << "H " << pkt.flow().str() << endl;
    Packet* pkt_p = &pkt;
    _queues[QUEUE_HIGH].push(pkt_p);
    _queuesize[QUEUE_HIGH] += pkt.size();
    if (_logger) _logger->logQueue(*this, QueueLogger::PKT_ENQUEUE, pkt);
    
    //cout << "BH[ " << _enqueued_low.size() << " " << _enqueued_high.size() << " ]" << endl;
    
    if (_serv==QUEUE_INVALID_VC) {
        beginService();
    }
}

mem_b CompositeQueue::queuesize() const {
    return _queuesize[QUEUE_HIGH] + _queuesize[QUEUE_LOW_VC_0] + _queuesize[QUEUE_LOW_VC_1];
}

void CompositeQueue::log_packet_send(simtime_picosec duration, int pkt_vc) {
    simtime_picosec b = eventlist().now();
    simtime_picosec a = b - duration;
    _vc_busystart[pkt_vc].push(a);
    _vc_busyend[pkt_vc].push(b);

    _vc_busy[pkt_vc] += duration;

    simtime_picosec y = _vc_busyend[pkt_vc].back();
    while (y < b - _window) {
        simtime_picosec x = _vc_busystart[pkt_vc].pop();
        _vc_busyend[pkt_vc].pop();

        _vc_busy[pkt_vc] -= (y-x);

        if (!_vc_busyend[pkt_vc].empty())
            y = _vc_busyend[pkt_vc].back();
        else
            break;
    }
}

uint16_t CompositeQueue::average_utilization(int pkt_vc) {
    if (_vc_busystart[pkt_vc].empty())
        return 0;

    simtime_picosec y = _vc_busyend[pkt_vc].back();
    simtime_picosec b = eventlist().now();

    while (y < b - _window) {
        simtime_picosec x = _vc_busystart[pkt_vc].pop();
        _vc_busyend[pkt_vc].pop();

        _vc_busy[pkt_vc] -= (y-x);
        assert(_vc_busy[pkt_vc] >= 0);

        if (!_vc_busyend[pkt_vc].empty())
            y = _vc_busyend[pkt_vc].back();
        else
            break;

    }
    return (_vc_busy[pkt_vc] * 100 / _window);
}

uint8_t CompositeQueue::quantized_utilization(int pkt_vc) {
    if (eventlist().now() - _vc_last_update_utilization[pkt_vc] > _update_period){
        _vc_last_update_utilization[pkt_vc] = eventlist().now();

        uint16_t avg = average_utilization(pkt_vc);

        if (avg == 0)
            _vc_last_utilization[pkt_vc] = 0;
        else if (avg < 15)
            _vc_last_utilization[pkt_vc] = 1;
        else if (avg < 50)
            _vc_last_utilization[pkt_vc] = 2;
        else
            _vc_last_utilization[pkt_vc] = 3;

    }
    return _vc_last_utilization[pkt_vc];
}

uint64_t CompositeQueue::quantized_queuesize(int pkt_vc) {
    if (eventlist().now()-_vc_last_update_qs[pkt_vc] > _update_period){
        _vc_last_update_qs[pkt_vc] = eventlist().now();

        uint64_t qs = _queuesize[pkt_vc];
        if (qs < _maxsize * 0.05)
            _vc_last_qs[pkt_vc] = 0;
        else if (qs < _maxsize * 0.1)
            _vc_last_qs[pkt_vc] = 1;
        else if (qs < _maxsize * 0.2)
            _vc_last_qs[pkt_vc] = 2;
        else
            _vc_last_qs[pkt_vc] = 3;
    }
    return _vc_last_qs[pkt_vc];
}
