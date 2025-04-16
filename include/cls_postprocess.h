#ifndef CLS_POSTPROCESS_H
#define CLS_POSTPROCESS_H
#include <vector>

namespace cls {
    using namespace std;

    struct Prob{
        int class_label;
        float confidence;

        Prob() = default;

        Prob(int class_label, float confidence)
        :class_label(class_label), confidence(confidence){}
    };
    typedef vector<Prob> ProbArray;

}

#endif //CLS_POSTPROCESS_H
