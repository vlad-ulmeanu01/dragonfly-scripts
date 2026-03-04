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
    std::map<std::pair<std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>, Greedy2par::Info> ht;
    std::vector<std::pair<int, int>> offsets(need_random.size());

    for (const auto& [packet, begin, end]: need_random) {
        auto p = std::make_pair(begin, end);
        if (ht.find(p) == ht.end()) ht[p] = Greedy2par::Info(std::distance(begin, end));
    }

    int i = 0;
    for (const auto& [packet, begin, end]: need_random) {
        auto p = std::make_pair(begin, end);

        offsets[i] = std::make_pair(std::uniform_int_distribution<int>(0, ht[p].dist-1)(mt), std::uniform_int_distribution<int>(0, ht[p].dist-1)(mt));
        ht[p].freq[offsets[i].first]++;
        ht[p].freq[offsets[i].second]++;
        ht[p].cnt++;

        i++;
    }

    std::vector<std::vector<NeighInfo>::iterator> its(need_random.size());
    int solved_packs = 0;
    
    while (solved_packs < (int)need_random.size()) {
        i = 0;
        for (const auto& [packet, begin, end]: need_random) {
            if (offsets[i].first >= 0) { ///nerezolvat pana acum.
                auto p = std::make_pair(begin, end);

                int qsz_fi = (begin + offsets[i].first)->out_qu.size(), qsz_se = (begin + offsets[i].second)->out_qu.size();
                int thresh = (ht[p].cnt + ht[p].dist - 1) / ht[p].dist;

                if (std::min(qsz_fi, qsz_se) <= thresh) {
                    if ((std::max(qsz_fi, qsz_se) <= thresh && qsz_fi <= qsz_se) || qsz_fi <= thresh) its[i] = begin + offsets[i].first;
                    else its[i] = begin + offsets[i].second;

                    offsets[i].first = -1;
                    solved_packs++;
                }
            }

            i++;
        }

        for (auto& x: ht) std::fill(x.second.freq.begin(), x.second.freq.end(), 0);

        if (solved_packs < (int)need_random.size()) {
            int i = 0;
            for (const auto& [packet, begin, end]: need_random) {
                if (offsets[i].first >= 0) {
                    auto p = std::make_pair(begin, end);
                    
                    offsets[i] = std::make_pair(std::uniform_int_distribution<int>(0, ht[p].dist-1)(mt), std::uniform_int_distribution<int>(0, ht[p].dist-1)(mt));
                    ht[p].freq[offsets[i].first]++;
                    ht[p].freq[offsets[i].second]++;
                }

                i++;
            }
        }
    }

    for (i = 0; i < (int)need_random.size(); i++) its[i]->out_qu.push(std::get<Packet>(need_random[i]));
}
