#include "automaton.hh"
#include "util/usage.hh"
#include <iomanip>
#include <limits>

namespace {
void CheckEqual(const lm::FullScoreReturn& lhs, const lm::FullScoreReturn& rhs) {
#ifdef NDEBUG
#define AUTOMATON_NDEBUG_WAS_SET
#undef NDEBUG
#endif
    assert(lhs.prob == rhs.prob);
    assert(lhs.independent_left == rhs.independent_left);
    assert(lhs.ngram_length == rhs.ngram_length);
    assert(lhs.rest == rhs.rest);
#ifdef AUTOMATON_NDEBUG_WAS_SET
#define NDEBUG
#undef AUTOMATON_NDEBUG_WAS_SET
#endif
}
}

struct Config {
    int pipeline_size_start;
    int pipeline_size_end;
    char* model_file;
    std::string type;
    int fd_in;
};

template <typename Callback>
void PipelineScore(lm::Pipeline<Callback>& pipeline, const lm::ngram::ProbingModel& model, const Config& options){
    const lm::WordIndex kEOS = model.GetVocabulary().EndSentence();
    const lm::ngram::State begin_state = model.BeginSentenceState();
    std::array<lm::WordIndex, 49806> buff;
    util::SeekOrThrow(options.fd_in, 0);

    //start timer
    auto time = util::CPUTime();
    std::size_t overhang = 0;
    while (true) {
        std::size_t got = util::ReadOrEOF(options.fd_in, buff.begin() + overhang, (buff.size() - overhang) * sizeof(lm::WordIndex));
        if (!got) break;
        UTIL_THROW_IF2(got % sizeof(lm::WordIndex), "File size not a multiple of vocab id size " << sizeof(lm::WordIndex));
        auto curr = buff.begin();
        auto end = curr + overhang + (got / sizeof(lm::WordIndex));
        auto sentence_begin = curr;
        while(curr != end) {
            if (*curr++ == kEOS) {
                pipeline.AppendWords(begin_state, sentence_begin, curr);
                sentence_begin = curr;
            }
        }
        UTIL_THROW_IF2(sentence_begin - buff.begin() == 0, "Buffer is too small");
        //Copy unused words
        std::copy(sentence_begin, end, buff.begin());
        overhang = end-sentence_begin;
        //std::cout << "Overhang" << overhang << '\n';
    }
    pipeline.Drain();
    //stop timer
    time = util::CPUTime() - time;
    std::cout << time << ' ';
}

void ModelScore(const lm::ngram::ProbingModel& model, const Config& options){
    const lm::WordIndex kEOS = model.GetVocabulary().EndSentence();
    const lm::ngram::State* const begin_state = &model.BeginSentenceState();
    const lm::ngram::State *in_state = begin_state;
    std::array<lm::WordIndex, 49806> buff;
    lm::ngram::State states[3];
    long double score = 0.0;

    //start timer
    auto time = util::CPUTime();
    bool new_sentence = true;
    while (true) {
      std::size_t got = util::ReadOrEOF(options.fd_in, buff.begin(), buff.size() * sizeof(lm::WordIndex));
      if (!got) break;
      UTIL_THROW_IF2(got % sizeof(lm::WordIndex), "File size not a multiple of vocab id size " << sizeof(lm::WordIndex));
      auto even_end = buff.begin() + ((got / sizeof(lm::WordIndex)) & ~1);
      auto curr = buff.begin();
      while(curr != even_end) {
        score += model.FullScore(*in_state, *curr, states[1]).prob;
        in_state = (*curr++ == kEOS) ? begin_state : &states[1];
        score += model.FullScore(*in_state, *curr, states[0]).prob;
        in_state = (*curr++ == kEOS) ? begin_state : &states[0];
      }
      if (got & 1) {
          score += model.FullScore(*in_state, *curr, states[2]).prob;
          in_state = *curr == kEOS ? begin_state : &states[2];
      }
    }
    //stop timer
    time = util::CPUTime() - time;
    std::cout << time << ' ';
    std::cerr << "Score(model) : " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << score << std::endl;
}

void DispatchFunction(lm::ngram::ProbingModel& model, const Config& options){
    if (options.type == "probing") ModelScore(model, options);
    else if (options.type == "pipeline") {
        long double score = 0.0;
        const auto callback = [&score](const lm::FullScoreReturn& r){score += r.prob;};
        typename lm::ngram::NGramAutomaton<lm::ngram::BackoffValue, decltype(callback)>::Construct construct{model.GetSearch(), callback};
        for (std::size_t pipeline_size = options.pipeline_size_start; pipeline_size <= options.pipeline_size_end; ++pipeline_size) {
            score = 0.0;
            lm::Pipeline<decltype(callback)> pipeline(pipeline_size, construct);
            PipelineScore<decltype(callback)>(pipeline, model, options);
            std::cerr << "Score(pipeline): " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << score << std::endl;
        }
    }
}



int main(int argc, char* argv[]){
    if (argc < 6) {
        std::cerr << argv[0] <<" pipeline_size_start pipeline_size_end model_file query_file {probing|pipeline}" << std::endl;
        return 1;
    }
    int pipeline_size_start = std::stoi(std::string(argv[1]));
    int pipeline_size_end = std::stoi(std::string(argv[2]));
    char* model_file(argv[3]);
    util::scoped_fd in_fd(util::OpenReadOrThrow(argv[4]));
    std::string type(argv[5]);
    Config options{pipeline_size_start, pipeline_size_end, model_file, type, in_fd.get()};

    lm::ngram::Config config;
    config.arpa_complain = lm::ngram::Config::ALL;
    config.messages = &std::cout;
    config.positive_log_probability = lm::SILENT;
    config.probing_multiplier = 1.5;
    lm::ngram::ProbingModel model(options.model_file, config);

    DispatchFunction(model, options);
}
