// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef DRAGON_FLY_PLUS
#define DRAGON_FLY_PLUS
#include "main.h"
#include "randomqueue.h"
#include "pipe.h"
#include "config.h"
#include "loggers.h"
#include "network.h"
#include "topology.h"
#include "logfile.h"
#include "eventlist.h"
#include "switch.h"
#include <ostream>

//Dragon Fly Plus parameters
// p = number of hosts per leaf router.
// l = number of leaf routers per group.
// s = number of spine routers per group.
// k = router radix.
// h = number of links from a spine router to other groups.
// g = number of groups.
// 
// Recommended is p = l = s = h = k / 2
// https://www.researchgate.net/publication/313341364_Dragonfly_Low_Cost_Topology_for_Scaling_Datacenters
//
// N_group = p * l = k^2 / 4
// N = k^4 / 16 + k^2 / 4 = p * l * (s * h + 1)
//
// SPARSE topology -> only 1 min path between Groups                         scales with k^4        (k^2)/4 * ((k^2)/4 + 1)
// DENSE topology  -> s (number of spine routers) min paths between Groups   scales with k^3        (k^2)/4 * (k/2 + 1)
//
////// Sparse global connections
// Groups are connected with this formula (named consecutive or absolute global link arrangement)
//      other possible types of global connections between groups could be added in the future
//          circulant based
//          
// if (srcgroup < dstgroup){
//     srcswitch = srcgroup * _s + (dstgroup-1)/_h;
//     dstswitch = dstgroup * _s + srcgroup/_h;
// }
// else {
//     srcswitch = srcgroup * _s + dstgroup/_h;
//     dstswitch = dstgroup * _s + (srcgroup-1)/_h;
// }
//

#ifndef QT
#define QT
typedef enum {UNDEFINED, RANDOM, ECN, COMPOSITE, PRIORITY,
              CTRL_PRIO, FAIR_PRIO, LOSSLESS, LOSSLESS_INPUT, LOSSLESS_INPUT_ECN,
              COMPOSITE_ECN, COMPOSITE_ECN_LB, SWIFT_SCHEDULER, ECN_PRIO, AEOLUS, AEOLUS_ECN} queue_type;
typedef enum {UPLINK, DOWNLINK} link_direction;
#endif

class DragonFlyPlusTopology: public Topology {
public:
    vector <Switch*> leafs;
    vector <Switch*> spines;

    vector< vector<Pipe*> > pipes_host_leaf;
    vector< vector<Pipe*> > pipes_leaf_spine;
    vector< vector<Pipe*> > pipes_spine_spine;
    vector< vector<Pipe*> > pipes_spine_leaf;
    vector< vector<Pipe*> > pipes_leaf_host;

    vector< vector<Queue*> > queues_host_leaf;
    vector< vector<Queue*> > queues_leaf_spine;
    vector< vector<Queue*> > queues_spine_spine;
    vector< vector<Queue*> > queues_spine_leaf;
    vector< vector<Queue*> > queues_leaf_host;
  
    QueueLoggerFactory* _logger_factory;
    EventList* _eventlist;
    uint32_t failed_links;
    queue_type qt;


    DragonFlyPlusTopology(uint32_t p, uint32_t l, uint32_t s, uint32_t h, linkspeed_bps linkspeed, mem_b queuesize, QueueLoggerFactory* logger_factory,EventList* ev,queue_type q,simtime_picosec latency,simtime_picosec switch_latency);
    DragonFlyPlusTopology(uint32_t radix, linkspeed_bps linkspeed, mem_b queuesize, QueueLoggerFactory* logger_factory,EventList* ev,queue_type q, simtime_picosec latency, simtime_picosec switch_latency, topology_type type = DFP_DENSE_T, const char *topo_dfp_sparse_file = NULL);

    void init_network();
    virtual vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse);
    void add_switch_loggers(Logfile& log, simtime_picosec sample_period);

    Route* setup_uec_route(int host_nr) override;

    Queue* alloc_src_queue(QueueLogger* q);
    Queue* alloc_queue(QueueLogger* q, mem_b queuesize, link_direction dir, bool tor);
    Queue* alloc_queue(QueueLogger* q, linkspeed_bps speed, mem_b queuesize, link_direction dir, bool tor);

    void count_queue(Queue*);
    void print_path(std::ofstream& paths, uint32_t src, const Route* route);
    vector<uint32_t>* get_neighbours(uint32_t src) { return NULL;};
    uint32_t no_of_nodes() const {return _no_of_nodes;};
    uint32_t HOST_TOR(uint32_t src) {return src / _p;};
    uint32_t HOST_GROUP(uint32_t src) {return src / (_l*_p);};
    uint32_t LEAF_GROUP(uint32_t src) {return src / _l;};
    uint32_t SPINE_GROUP(uint32_t src) {return src / _s;};
    uint32_t getNHostsLeafs() {return _p * _l;};
    uint32_t getNLeafsGroup() {return _l;};
    uint32_t getNSpinesGroup() {return _s;};
    uint32_t getNGroups() {return _no_of_groups;};
    uint32_t getNGlobalLinks() {return _h;};
    uint32_t getTopologyType() {return _type;};
    int get_oversubscription_ratio() {return get_oversubscription_ratio(1);}; // Basically route_strategy=Switch::ECMP
    int get_oversubscription_ratio(uint32_t route_strategy);
    simtime_picosec get_diameter_latency() {return 8 * _hop_latency + 7 * _switch_latency;};
    simtime_picosec get_two_point_diameter_latency(int src, int dst);    
    void connectHostToHostQueue(uint32_t src, UecSrcPort *port_src) override;
    std::vector<std::vector<uint32_t>>& get_sparse_cfg_reference() { return _topo_dfp_sparse_cfg; }

private:
    int64_t find_switch(Queue* queue);
    int64_t find_destination(Queue* queue);

    void set_params(uint32_t no_of_nodes);
    void set_params();

    uint32_t _k, _p, _l, _s, _h;
    uint32_t _type;
    const char *_topo_dfp_sparse_file;
    std::vector<std::vector<uint32_t>> _topo_dfp_sparse_cfg;
    uint32_t _no_of_nodes;
    uint32_t _no_of_groups,_no_of_switches, _no_of_leafs, _no_of_spines;
    simtime_picosec _hop_latency, _switch_latency;
    mem_b _queuesize;
    linkspeed_bps _linkspeed;
};

#endif
