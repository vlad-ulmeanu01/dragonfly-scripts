// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef _DRAGONFLY_PLUS_SWITCH_H
#define _DRAGONFLY_PLUS_SWITCH_H

#include "switch.h"
#include "callback_pipe.h"

class DragonFlyPlusTopology;

class DragonFlyPlusSwitch : public Switch {
public:
    enum switch_type {
        NONE = 0, LEAF = 1, SPINE = 2
    };

    DragonFlyPlusSwitch(EventList& eventlist, string s, switch_type t, uint32_t id,simtime_picosec switch_delay, DragonFlyPlusTopology* dfp);
  
    static int8_t compare_queuesize_dense(FibEntry* left, FibEntry* right);
    static int8_t compare_queuesize_sparse(FibEntry* left, FibEntry* right);
    virtual void receivePacket(Packet& pkt);
    virtual Route* getNextHop(Packet& pkt, BaseQueue* ingress_port);
    virtual uint32_t getType() {return _type;}

    uint32_t adaptive_route(vector<FibEntry*>* ecmp_set, int8_t (*cmp)(FibEntry*,FibEntry*));
    uint32_t replace_worst_choice(vector<FibEntry*>* ecmp_set, int8_t (*cmp)(FibEntry*,FibEntry*),uint32_t my_choice);
    uint32_t adaptive_route_p2c(vector<FibEntry*>* ecmp_set, int8_t (*cmp)(FibEntry*,FibEntry*));

    virtual void addHostPort(int addr, int flowid, PacketSink* transport_port);

    virtual void permute_paths(vector<FibEntry*>* uproutes);

private:
    switch_type _type;
    Pipe* _pipe;
    DragonFlyPlusTopology* _dfp;
    std::vector<std::vector<uint32_t>>& _topo_dfp_sparse_cfg;
    // int get_spine_id_linking_groups_in_cfg(int group_from, int group_to); /// TODO.
    
    //CAREFUL: can't always have a single FIB for all up destinations when there are failures!
    vector<FibEntry*>* _uproutes;

    unordered_map<uint32_t,FlowletInfo*> _flowlet_maps;

    uint32_t _crt_route;
    uint32_t _hash_salt;
    simtime_picosec _last_choice;

    unordered_map<Packet*,bool> _packets;
};

#endif
    
