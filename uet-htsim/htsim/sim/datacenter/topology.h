// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef TOPOLOGY
#define TOPOLOGY
#include "network.h"
#include "loggers.h"

class UecSrcPort;

#ifndef TOPOLOGY_TYPE
#define TOPOLOGY_TYPE
typedef enum {FAT_TREE_T = 0, DFP_DENSE_T = 1, DFP_SPARSE_T = 2} topology_type;
#endif

class Topology {
public:
    vector <Switch*> tors;

    virtual vector<const Route*>* get_paths(uint32_t src, uint32_t dest) {
        return get_bidir_paths(src, dest, true);
    }
    virtual vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse)=0;
    virtual vector<uint32_t>* get_neighbours(uint32_t src) = 0;  
    virtual uint32_t no_of_nodes() const {
        abort();
    }

    // add loggers to record total queue size at switches
    virtual void add_switch_loggers(Logfile& log, simtime_picosec sample_period) {
        abort();
    }
    virtual int get_oversubscription_ratio() {abort();};
    virtual int get_oversubscription_ratio(uint32_t route_strategy) {abort();};
    virtual uint16_t get_diameter() {return _diameter;}
    virtual simtime_picosec get_two_point_diameter_latency(int src, int dst) {abort();};
    virtual simtime_picosec get_diameter_latency() {abort();};

    virtual Route* setup_uec_route(int host_nr) {abort();};
    virtual uint32_t HOST_TOR(uint32_t src) {abort();};

    virtual void connectHostToHostQueue(uint32_t src, UecSrcPort *port_src) {abort();};

    virtual ~Topology() = default;

    static bool _enable_ecn;
    static bool _enable_ecn_on_tor_downlink;
    static mem_b _ecn_low;
    static mem_b _ecn_high;

    // failed links hack
    static uint32_t _num_failed_links;
    static double _failed_link_ratio;
    uint16_t _diameter;
};

#endif
