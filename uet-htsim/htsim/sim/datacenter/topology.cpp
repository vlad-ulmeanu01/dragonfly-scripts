#include "topology.h"

bool Topology::_enable_ecn = false;
bool Topology::_enable_ecn_on_tor_downlink = false;
mem_b Topology::_ecn_low = 0;
mem_b Topology::_ecn_high = 0;
uint32_t Topology::_num_failed_links = 0;
double Topology::_failed_link_ratio = 0.25;
