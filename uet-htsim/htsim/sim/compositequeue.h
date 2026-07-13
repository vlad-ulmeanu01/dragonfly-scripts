// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        
#ifndef COMPOSITE_QUEUE_H
#define COMPOSITE_QUEUE_H

/*
 * A composite queue that transforms packets into headers when there is no space and services headers with priority. 
 */

#define QUEUE_INVALID_VC -1
#define QUEUE_LOW_VC_0 0
#define QUEUE_LOW_VC_1 1
#define QUEUE_HIGH     2


#include <list>
#include "queue.h"
#include "config.h"
#include "eventlist.h"
#include "network.h"
#include "eth_pause_packet.h"
#include "loggertypes.h"

#define VCs 3
#define QUEUE_HIGH 2

class CompositeQueue : public Queue {
 public:
    CompositeQueue(linkspeed_bps bitrate, mem_b maxsize, 
                   EventList &eventlist, QueueLogger* logger, 
                   uint16_t trim_size, bool disable_trim=false);
    virtual void receivePacket(Packet& pkt);
    virtual void doNextEvent();

    enum queue_state {PAUSED,READY,PAUSE_RECEIVED};

    // should really be private, but loggers want to see
    mem_b _queuesize[VCs] = {0};
    queue_state _states[VCs] = {READY, READY, READY};
    int num_headers() const { return _num_headers;}
    int num_packets() const { return _num_packets;}
    int num_stripped() const { return _num_stripped;}
    int num_bounced() const { return _num_bounced;}
    int num_acks() const { return _num_acks;}
    int num_nacks() const { return _num_nacks;}
    int num_pulls() const { return _num_pulls;}
    mem_b queuesize_high_watermark() const { return _queuesize_high_watermark;}
    virtual mem_b queuesize() const;
    virtual void setName(const string& name) {
        Logged::setName(name); 
        _nodename += name;
    }

    void setRTS(bool return_to_sender){ _return_to_sender = return_to_sender;}

    virtual const string& nodename() { return _nodename; }
    void set_ecn_threshold(mem_b ecn_thresh) {
        _ecn_minthresh = ecn_thresh;
        _ecn_maxthresh = ecn_thresh;
    }
    void set_ecn_thresholds(mem_b min_thresh, mem_b max_thresh) {
        _ecn_minthresh = min_thresh;
        _ecn_maxthresh = max_thresh;
        if (_queue_id == 2)
            cout << "queue_id " << _queue_id << " ecn_low " << _ecn_minthresh << " ecn_high " << _ecn_maxthresh << endl;
    }

    bool is_paused(int pkt_tc = 0) { return _states[pkt_tc] == PAUSED || _states[pkt_tc] == PAUSE_RECEIVED;}

    void log_packet_send(simtime_picosec duration, int pkt_vc);
    uint16_t average_utilization(int pkt_vc);
    uint8_t quantized_utilization(int pkt_vc);
    uint64_t quantized_queuesize(int pkt_vc);

    int _num_packets;
    int _num_headers; // only includes data packets stripped to headers, not acks or nacks
    int _num_acks;
    int _num_nacks;
    int _num_pulls;
    int _num_stripped; // count of packets we stripped
    int _num_bounced;  // count of packets we bounced
    mem_b _queuesize_high_watermark; // max occupancy of high priority queue
    static bool isLossless;

 protected:
    // Mechanism
    void beginService(); // start serving the item at the head of the queue
    void completeService(); // wrap up serving the item at the head of the queue
    bool decide_ECN();
    void processPause(EthPausePacket* p);

    bool _disable_trim;

    int _serv;
    int _ratios[VCs];
    int _crts[VCs] = {0};
    // below minthresh, 0% marking, between minthresh and maxthresh
    // increasing random mark propbability, abve maxthresh, 100%
    // marking.
    mem_b _ecn_minthresh; 
    mem_b _ecn_maxthresh;

    uint16_t _trim_size;

    bool _return_to_sender;

    CircularBuffer<simtime_picosec> _vc_busystart[VCs];
    CircularBuffer<simtime_picosec> _vc_busyend[VCs];
    simtime_picosec _vc_busy[VCs] = {0};
    simtime_picosec _vc_last_update_qs[VCs] = {0};
    simtime_picosec _vc_last_update_utilization[VCs] = {0};
    uint8_t _vc_last_utilization[VCs] = {0};
    uint8_t _vc_last_qs[VCs] = {0};

    int _queue_id;
    CircularBuffer<Packet*> _queues[VCs];
};

#endif
