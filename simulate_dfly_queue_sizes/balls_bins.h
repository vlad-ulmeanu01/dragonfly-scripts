#include "utils.h"

struct BallsBins {
    std::array<Node, DFLY_SIZE>& dfly;
    std::random_device rd;
    std::mt19937 mt;

    BallsBins(std::array<Node, DFLY_SIZE>& dfly): dfly(dfly), mt(DEBUG? 0: rd()) {}

    ///pentru fiecare pachet trebuie ales un iterator din [begin, end) in a carei coada o sa-l bagam.
    ///pentru Greedy1/2(seq/par) conteaza doar packets.size(), nu si continutul.
    virtual void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};

struct Greedy1: BallsBins {
    Greedy1(std::array<Node, DFLY_SIZE>& dfly): BallsBins(dfly) {}

    void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};

struct Greedy2seq: BallsBins {
    Greedy2seq(std::array<Node, DFLY_SIZE>& dfly): BallsBins(dfly) {}

    void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};

struct Greedy2par: BallsBins {
    Greedy2par(std::array<Node, DFLY_SIZE>& dfly): BallsBins(dfly) {}

    void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};
