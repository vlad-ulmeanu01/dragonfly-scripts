// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        
#include "queue_lossless_input.h"
#include <math.h>
#include <iostream>
#include <sstream>
#include "switch.h"
#include "compositequeue.h"

uint64_t LosslessInputQueue::_high_threshold = 0;
uint64_t LosslessInputQueue::_low_threshold = 0;

LosslessInputQueue::LosslessInputQueue(EventList& eventlist)
    : Queue(speedFromGbps(1),Packet::data_packet_size()*2000,eventlist,NULL),
      VirtualQueue()
{
    if (CompositeQueue::isLossless) {
        assert(_high_threshold>0);
        assert(_high_threshold > _low_threshold);
    }

    _wire = NULL;
}

LosslessInputQueue::LosslessInputQueue(EventList& eventlist,BaseQueue* peer)
    : Queue(speedFromGbps(1),Packet::data_packet_size()*2000,eventlist,NULL),
      VirtualQueue()
{
    if (CompositeQueue::isLossless) {
        assert(_high_threshold>0);
        assert(_high_threshold > _low_threshold);
    }

    stringstream ss;
    ss << "VirtualQueue("<< peer->_name<< ")";
    _nodename = ss.str();
    _remoteEndpoint = peer;
    _switch = NULL;
    _wire = NULL;

    peer->setRemoteEndpoint(this);
}

LosslessInputQueue::LosslessInputQueue(EventList& eventlist,BaseQueue* peer, Switch* sw, simtime_picosec wire_latency)
    : Queue(speedFromGbps(1),Packet::data_packet_size()*2000,eventlist,NULL),
      VirtualQueue()
{
    if (CompositeQueue::isLossless) {
        assert(_high_threshold>0);
        assert(_high_threshold > _low_threshold);
    }

    stringstream ss;
    ss << "VirtualQueue("<< peer->_name<< ")";
    _nodename = ss.str();
    _remoteEndpoint = peer;
    _switch = sw;

    _wire = new CallbackPipe(wire_latency, eventlist, _remoteEndpoint);

    assert(_switch);

    peer->setRemoteEndpoint(this);
}


void
LosslessInputQueue::receivePacket(Packet& pkt) 
{
    /* normal packet, enqueue it */
    int pkt_vc = pkt.vc();
    _vc_quesize[pkt_vc] += pkt.size();

    //send PAUSE notifications if that is the case!
    assert(_vc_quesize[pkt_vc] > 0);
    if (CompositeQueue::isLossless && (uint64_t)_vc_quesize[pkt_vc] > _high_threshold && _vc_state[pkt_vc] != PAUSED){
        _vc_state[pkt_vc] = PAUSED;
        sendPause(1000, pkt_vc);
    }

    //if (_state_recv==PAUSED)
    //cout << timeAsMs(eventlist().now()) << " queue " << _name << " switch (" << _switch->_name << ") "<< " recv when paused pkt " << pkt.type() << " sz " << _queuesize << endl;        

    if (CompositeQueue::isLossless && _vc_quesize[pkt_vc] > _maxsize){
        cout << " Queue " << _nodename << " LOSSLESS not working! I should have dropped this packet" << _vc_quesize[pkt_vc] / Packet::data_packet_size() << endl;
    }
    
    //tell the output queue we're here!
    if (pkt.nexthop() < pkt.route()->size()){
        //this should not work...
        //assert(0);
        pkt.sendOn2(this);
    }
    else {
        assert(_switch);
        pkt.set_ingress_queue(this);
        _switch->receivePacket(pkt);
    }
}

void LosslessInputQueue::completedService(Packet& pkt) {
    int pkt_vc = pkt.vc();
    if (pkt._justChangedVC)
        pkt_vc = pkt.prevVc();
    _vc_quesize[pkt_vc] -= pkt.size();

    //unblock if that is the case
    assert(_vc_quesize[pkt_vc] >= 0);
    if (CompositeQueue::isLossless && (uint64_t)_vc_quesize[pkt_vc] < _low_threshold && _vc_state[pkt_vc] == PAUSED) {
        _vc_state[pkt_vc] = READY;
        sendPause(0, pkt_vc);
    }
}

void LosslessInputQueue::sendPause(unsigned int wait, int vc){
    //cout << "Ingress link " << getRemoteEndpoint() << " PAUSE " << wait << endl;    
    uint32_t switchID = 0;
    if (_switch)
        switchID = getSwitch()->getID();

    EthPausePacket* pkt = EthPausePacket::newpkt(wait,switchID,vc);

    if (_wire)
        _wire->receivePacket(*pkt);
    else
        getRemoteEndpoint()->receivePacket(*pkt);
};
