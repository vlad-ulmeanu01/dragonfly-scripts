// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        

#include "queue.h"
#include "switch.h"
#include "eth_pause_packet.h"
#include "queue_lossless.h"
#include "queue_lossless_input.h"
#include "queue_lossless_output.h"
#include "loggers.h"

uint32_t Switch::id = 0;
unordered_map<BaseQueue*,uint32_t> Switch::_port_flow_counts;
Switch::routing_strategy Switch::_strategy = Switch::NIX;
uint16_t Switch::_ar_fraction = 0;
uint16_t Switch::_ar_sticky = Switch::PER_PACKET;
simtime_picosec Switch::_sticky_delta = timeFromUs((uint32_t)10);
double Switch::_ecn_threshold_fraction = 0.2;
double Switch::_speculative_threshold_fraction = 0.2;
uint16_t Switch::_trim_size = 64;
int8_t (*Switch::fn)(FibEntry*,FibEntry*) = &Switch::compare_queuesize;
bool Switch::_disable_trim = false;

int Switch::addPort(BaseQueue* q){
    _ports.push_back(q);
    q->setSwitch(this);
    return _ports.size()-1;
    
}

void Switch::sendPause(LosslessQueue* problem, unsigned int wait){
    cout << "Switch " << _name << " link " << problem->_name << " blocked, sending pause " << wait << endl;

    for (size_t i = 0;i < _ports.size();i++){
        LosslessQueue* q = (LosslessQueue*)_ports.at(i);
    
        if (q==problem || !(q->getRemoteEndpoint()))
            continue;

        cout << "Informing " << q->_name << endl;
        EthPausePacket* pkt = EthPausePacket::newpkt(wait,_id);
        q->getRemoteEndpoint()->receivePacket(*pkt);
    }
};

void Switch::configureLossless(){
    for (size_t i = 0;i < _ports.size();i++){
        LosslessQueue* q = (LosslessQueue*)_ports.at(i);    
        q->setSwitch(this);
        q->initThresholds();
    }
};
/*Switch::configureLosslessInput(){
  for (list<Queue*>::iterator it=_ports.begin(); it != _ports.end(); ++it){
  LosslessInputQueue* q = (LosslessInputQueue*)*it;
  q->setSwitch(this);
  q->initThresholds();
  }
  };*/

int8_t Switch::compare_pause(FibEntry* left, FibEntry* right){
    Route * r1= left->getEgressPort();
    assert(r1 && r1->size()>1);
    LosslessOutputQueue* q1 = dynamic_cast<LosslessOutputQueue*>(r1->at(0));
    Route * r2= right->getEgressPort();
    assert(r2 && r2->size()>1);
    LosslessOutputQueue* q2 = dynamic_cast<LosslessOutputQueue*>(r2->at(0));

    if (!q1->is_paused()&&q2->is_paused())
        return 1;
    else if (q1->is_paused()&&!q2->is_paused())
        return -1;
    else 
        return 0;
}

int8_t Switch::compare_flow_count(FibEntry* left, FibEntry* right){
    Route * r1= left->getEgressPort();
    assert(r1 && r1->size()>1);
    BaseQueue* q1 = (BaseQueue*)(r1->at(0));
    Route * r2= right->getEgressPort();
    assert(r2 && r2->size()>1);
    BaseQueue* q2 = (BaseQueue*)(r2->at(0));

    if (_port_flow_counts.find(q1)==_port_flow_counts.end())
        _port_flow_counts[q1] = 0;

    if (_port_flow_counts.find(q2)==_port_flow_counts.end())
        _port_flow_counts[q2] = 0;

    //cout << "CMP q1 " << q1 << "=" << _port_flow_counts[q1] << " q2 " << q2 << "=" << _port_flow_counts[q2] << endl; 

    if (_port_flow_counts[q1] < _port_flow_counts[q2])
        return 1;
    else if (_port_flow_counts[q1] > _port_flow_counts[q2] )
        return -1;
    else 
        return 0;
}

int8_t Switch::compare_queuesize(FibEntry* left, FibEntry* right){
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

int8_t Switch::compare_bandwidth(FibEntry* left, FibEntry* right){
    Route * r1= left->getEgressPort();
    assert(r1 && r1->size()>1);
    BaseQueue* q1 = dynamic_cast<BaseQueue*>(r1->at(0));
    Route * r2= right->getEgressPort();
    assert(r2 && r2->size()>1);
    BaseQueue* q2 = dynamic_cast<BaseQueue*>(r2->at(0));

    if (q1->quantized_utilization() < q2->quantized_utilization())
        return 1;
    else if (q1->quantized_utilization() > q2->quantized_utilization())
        return -1;
    else 
        return 0;

    /*if (q1->average_utilization() < q2->average_utilization())
        return 1;
    else if (q1->average_utilization() > q2->average_utilization())
        return -1;
    else 
        return 0;        */
}

int8_t Switch::compare_pqb(FibEntry* left, FibEntry* right){
    //compare pause, queuesize, bandwidth.
    int8_t p = compare_pause(left, right);

    if (p!=0)
        return p;
    
    p = compare_queuesize(left,right);

    if (p!=0)
        return p;

    return compare_bandwidth(left,right);
}

int8_t Switch::compare_pq(FibEntry* left, FibEntry* right){
    //compare pause, queuesize, bandwidth.
    int8_t p = compare_pause(left, right);

    if (p!=0)
        return p;
    
    return compare_queuesize(left,right);
}

int8_t Switch::compare_qb(FibEntry* left, FibEntry* right){
    //compare pause, queuesize, bandwidth.
    int8_t p = compare_queuesize(left, right);

    if (p!=0)
        return p;
    
    return compare_bandwidth(left,right);
}

int8_t Switch::compare_pb(FibEntry* left, FibEntry* right){
    //compare pause, queuesize, bandwidth.
    int8_t p = compare_pause(left, right);

    if (p!=0)
        return p;
    
    return compare_bandwidth(left,right);
}

void Switch::add_logger(Logfile& log, simtime_picosec sample_period) {
    // we want to log the sum of all queues on the switch, so we have
    // one logger that is shared by all ports
    assert(_ports.size() > 0);
    MultiQueueLoggerSampling* queue_logger = new MultiQueueLoggerSampling(get_id(), sample_period,_ports.at(0)->eventlist());
    log.addLogger(*queue_logger);
    for (size_t i = 0; i < _ports.size(); i++) {
        //cout << "adding logger to switch " << nodename() << " id " << get_id() << " queue " << _ports.at(i)->nodename() << " id " << _ports.at(i)->get_id() << endl;
        _ports.at(i)->setLogger(queue_logger);
    }
}
