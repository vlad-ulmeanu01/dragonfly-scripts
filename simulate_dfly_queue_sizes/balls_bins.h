#include "utils.h"

struct BallsBins {
    DflyPlusMaxHosts& dfly;
    std::random_device rd;
    std::mt19937 mt;

    BallsBins(DflyPlusMaxHosts& dfly);
    virtual ~BallsBins() = default;

    ///pentru fiecare pachet trebuie ales un iterator din [begin, end) in a carei coada o sa-l bagam.
    ///pentru Greedy1/2(seq/par) conteaza doar packets.size(), nu si continutul.
    virtual void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    ) = 0;
};

struct Greedy1: BallsBins {
    Greedy1(DflyPlusMaxHosts& dfly);

    void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};

struct Greedy2seq: BallsBins {
    Greedy2seq(DflyPlusMaxHosts& dfly);

    void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};

struct Greedy2par: BallsBins {
    Greedy2par(DflyPlusMaxHosts& dfly);

    struct Info {
        std::vector<int> freq; ///frecventa alegerilor (pe care vrem sa le facem)
        std::vector<int> freq_chosen, freq_chosen_delta; /// frecventa pentru bilele pe care le-am ales dupa primele ?? runde.
        int dist, cnt;
        
        Info(int dist): freq(dist), freq_chosen(dist), freq_chosen_delta(dist), dist(dist), cnt(0) {}
        Info(): dist(0), cnt(0) {}
    };

    void make_picks(
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>>& need_random
    );
};
