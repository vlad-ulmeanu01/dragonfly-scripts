#include "balls_bins.h"

void BallsBins::make_picks(
    std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
) {}

void Greedy1::make_picks(
    std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
) {
    for (const auto& [packet, begin, end]: need_random) {
        int dist = std::distance(begin, end);
        auto it = begin + std::uniform_int_distribution<int>(0, dist-1)(mt);
        it->out_qu.push(packet);
    }
}

void Greedy2seq::make_picks(
    std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
) {
    std::shuffle(need_random.begin(), need_random.end(), mt);

    for (const auto& [packet, begin, end]: need_random) {
        int dist = std::distance(begin, end);
        auto it = begin + std::uniform_int_distribution<int>(0, dist-1)(mt), it2 = begin + std::uniform_int_distribution<int>(0, dist-1)(mt);
        if (it->out_qu.size() > it2->out_qu.size()) it = it2;
        it->out_qu.push(packet);
    }
}

void Greedy2par::make_picks(
    std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
) {
    std::vector<std::vector<NeighInfo>::iterator> its(need_random.size());
    int i = 0;
    for (const auto& [packet, begin, end]: need_random) {
        int dist = std::distance(begin, end);
        auto it = begin + std::uniform_int_distribution<int>(0, dist-1)(mt), it2 = begin + std::uniform_int_distribution<int>(0, dist-1)(mt);
        if (it->out_qu.size() > it2->out_qu.size()) it = it2;
        its[i++] = it;
    }

    for (i = 0; i < (int)need_random.size(); i++) its[i]->out_qu.push(std::get<Packet>(need_random[i]));
}
