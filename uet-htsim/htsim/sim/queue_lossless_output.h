// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        
#ifndef _LOSSLESS_OUTPUT_QUEUE_H
#define _LOSSLESS_OUTPUT_QUEUE_H
/*
 * A FIFO queue that supports PAUSE frames and lossless operation
 */

#include <list>
#include "queue.h"
#include "config.h"
#include "eventlist.h"
#include "network.h"
#include "loggertypes.h"
#include "eth_pause_packet.h"
#include "ecn.h"

class LosslessOutputQueue : public Queue {
public:
    LosslessOutputQueue(linkspeed_bps bitrate, mem_b maxsize, EventList &eventlist, QueueLogger* logger);

    void receivePacket(Packet& pkt);
    void receivePacket(Packet& pkt,VirtualQueue* q);

    void beginService();
    void completeService();

    bool is_paused() { return _state_send == PAUSED || _state_send == PAUSE_RECEIVED;}

    enum queue_state {PAUSED,READY,PAUSE_RECEIVED};

    void set_ecn_thresholds(mem_b min_thresh, mem_b max_thresh) {
        _ecn_minthresh = min_thresh;
        _ecn_maxthresh = max_thresh;
    }

private:
    list<VirtualQueue*> _vq;

    int _state_send;
    int _sending;
    uint64_t _txbytes;

    mem_b _ecn_minthresh; 
    mem_b _ecn_maxthresh;

    bool decide_ECN();

public:
    static int _ecn_enabled;
    static int _K;
    
};

#endif
