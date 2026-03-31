// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-        

#ifndef _SWITCH_H
#define _SWITCH_H
#include "queue.h"
/*
 * A switch to group together multiple ports (currently used in the
 * PAUSE implementation), and in generic_topology
 *
 * At the moment we don't normally build topologies where the switch
 * receives a packet and makes a forwarding decision - the route
 * already carries the forwarding path.  But we might revisit this to
 * simulate switches that make dynamic decisions.
 */

#include <list>
#include "config.h"
#include "eventlist.h"
#include "network.h"
#include "loggertypes.h"
#include "drawable.h"
#include "routetable.h"
#include <unordered_map>

class BaseQueue;
class LosslessQueue;
class LosslessInputQueue;
class RouteTable;
class FibEntry;

#define AR_BIT_nmin_1 8
#define AR_BIT_nmin_3 16

/*
 * Shamelessly copied from FreeBSD
 */

/* ----- FreeBSD if_bridge hash function ------- */

/*
 * The following hash function is adapted from "Hash Functions" by Bob Jenkins
 * ("Algorithm Alley", Dr. Dobbs Journal, September 1997).
 *
 * http://www.burtleburtle.net/bob/hash/spooky.html
 */

#define MIX(a, b, c)                            \
    do {                                        \
        a -= b; a -= c; a ^= (c >> 13);         \
        b -= c; b -= a; b ^= (a << 8);          \
        c -= a; c -= b; c ^= (b >> 13);         \
        a -= b; a -= c; a ^= (c >> 12);         \
        b -= c; b -= a; b ^= (a << 16);         \
        c -= a; c -= b; c ^= (b >> 5);          \
        a -= b; a -= c; a ^= (c >> 3);          \
        b -= c; b -= a; b ^= (a << 10);         \
        c -= a; c -= b; c ^= (b >> 15);         \
    } while (/*CONSTCOND*/0)

static inline uint32_t freeBSDHash(uint32_t target1, uint32_t target2 = 0, uint32_t target3 = 0)
{
    uint32_t a = 0x9e3779b9, b = 0x9e3779b9, c = 0; // hask key
        
    b += target3;
    c += target2;
    a += target1;        
    MIX(a, b, c);
    return c;
}

#undef MIX

class FlowletInfo {
public:
    uint32_t _egress;
    simtime_picosec _last;

    FlowletInfo(uint32_t egress,simtime_picosec lasttime) {_egress = egress; _last = lasttime;};

};

class Switch : public EventSource, public Drawable, public PacketSink {
 public:
    enum routing_strategy {
        NIX = 0, ECMP = 1, ADAPTIVE_ROUTING = 2, ECMP_ADAPTIVE = 3, RR = 4, RR_ECMP = 5, ECMP_ALL = 6
    };

    enum sticky_choices {
        PER_PACKET = 0, PER_FLOWLET = 1
    };

    Switch(EventList& eventlist) : EventSource(eventlist, "none") { _name = "none"; _id = id++;};
    Switch(EventList& eventlist, string s) : EventSource(eventlist, s) { _name= s; _id = id++;}
    virtual ~Switch() = default;

    virtual int addPort(BaseQueue* q);
    virtual void addHostPort(int addr, int flowid, PacketSink* transport) { abort();};

    uint32_t getID(){return _id;};
    virtual uint32_t getType() {return 0;}

    // inherited from PacketSink - only use when route strategy implies use of ECMP_FIB, i.e. the packet does not carry a full route. .
    virtual void receivePacket(Packet& pkt) {abort();}
    virtual void receivePacket(Packet& pkt,VirtualQueue* prev) {abort();}
    virtual void doNextEvent() {abort();}

    //used when route strategy is ECMP_FIB and variants. 
    virtual Route* getNextHop(Packet& pkt) { return getNextHop(pkt, NULL);}
    virtual Route* getNextHop(Packet& pkt, BaseQueue* ingress_port) {abort();};

    BaseQueue* getPort(int id) { assert(id >= 0); if ((unsigned int)id<_ports.size()) return _ports.at(id); else return NULL;}

    unsigned int portCount(){ return _ports.size();}

    void sendPause(LosslessQueue* problem, unsigned int wait);
    void sendPause(LosslessInputQueue* problem, unsigned int wait);

    void configureLossless();
    void configureLosslessInput();

    void add_logger(Logfile& log, simtime_picosec sample_period); 

    static int8_t compare_flow_count(FibEntry* l, FibEntry* r);
    static int8_t compare_pause(FibEntry* l, FibEntry* r);
    static int8_t compare_bandwidth(FibEntry* l, FibEntry* r);
    static int8_t compare_queuesize(FibEntry* l, FibEntry* r);
    static int8_t compare_pqb(FibEntry* l, FibEntry* r);//compare pause,queue, bw.
    static int8_t compare_pq(FibEntry* l, FibEntry* r);//compare pause, queue
    static int8_t compare_pb(FibEntry* l, FibEntry* r);//compare pause, bandwidth
    static int8_t compare_qb(FibEntry* l, FibEntry* r);//compare pause, bandwidth

    static void set_strategy(routing_strategy s) { assert (_strategy==NIX); _strategy = s; }
    static void set_ar_fraction(uint16_t f) { assert(f>=1);_ar_fraction = f; }

    virtual const string& nodename() {return _name;}

    static int8_t (*fn)(FibEntry*,FibEntry*);

    static routing_strategy _strategy;
    static uint16_t _ar_fraction;
    static uint16_t _ar_sticky;
    static simtime_picosec _sticky_delta;
    static double _ecn_threshold_fraction;
    static double _speculative_threshold_fraction;
    static uint16_t _trim_size;
    static bool _disable_trim;

protected:
    static unordered_map<BaseQueue*,uint32_t> _port_flow_counts;
    vector<BaseQueue*> _ports;
    uint32_t _id;
    string _name;

    RouteTable* _fib;
 
    static uint32_t id;
};
#endif
