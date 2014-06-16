#!/usr/bin/env ruby

require 'zipf'
require 'bloom-filter'


class FeatureFactory

  def initialize cfg
    @use_target_ngrams = false
    if cfg['ff_target_ngrams']
      @use_target_ngrams = true
      args = cfg['ff_target_ngrams'].split
      @target_ngrams_n = args[0].to_i
      @target_ngrams_fix = true if args.size==2&&args[1]=='fix'
    end
    @use_phrase_pairs = false
    if cfg['ff_phrase_pairs']
      @use_phrase_pairs = true
      @phrase_table = nil
      args = cfg['ff_phrase_pairs'].split
      if args.size==2
        @phrase_table = BloomFilter.load args.last
      end
    end
    @additional_phrase_pairs = {}
    @binary = false
    @binary = true if cfg['binary_feature_values']
    @filter_features = false
    if cfg['filter_features']
      @filter_features = true
      @stopwords_target = ReadFile.readlines(cfg['filter_features']).map{ |i| i.strip.downcase }
    end
  end

  def produce translation, source
    f = SparseVector.new
    phrase_pairs(f, translation, source) if @use_phrase_pairs
    target_ngrams(f, translation.s) if @use_target_ngrams
    return f
  end

  def filter a
    single_char = only_stop = only_num = 1
    a.each { |i|
      single_char = 0 if i.size > 1
      only_stop = 0 if not @stopwords_target.include? i.downcase
      only_num = 0 if i.gsub(/[0-9]+/, '').size > 0
    }
    return [single_char,only_stop,only_num].max==1
  end

  def phrase_pairs f, translation, source
    target_phrases = translation.get_phrases
    return if !target_phrases
    spans = translation.get_spans
    src_tok = source.split.map{ |i| i.strip }
    src_sz = 0.0
    name = nil
    spans.each_with_index { |i,j|
      next if @filter_features && filter(target_phrases[j])
      i.pop if i.size==2 && i[0]==i[1]
      if i.size == 2
        next if !src_tok[i[0]..i[1]]
        if @phrase_table
          pp = "#{src_tok[i[0]..i[1]].join ' '} ||| #{target_phrases[j]}"
          next if !(@phrase_table.include?(pp) || @additional_phrase_pairs.has_key?(pp))
        end
        name = "PP:#{src_tok[i[0]..i[1]].join ','}~#{target_phrases[j].split.join ','}"
        src_sz = src_tok[i[0]..i[1]].size.to_f
      else
        if @phrase_table
          pp ="#{src_tok[i[0]]} ||| #{target_phrases[j]}"
          next if !(@phrase_table.include?(pp) || @additional_phrase_pairs.has_key?(pp))
        end
        if i[0] >= 0
          name = "PP:#{src_tok[i[0]]}~#{target_phrases[j]}"
          src_sz = 1.0
        end
      end
      if @binary
        f[name] = 1.0
      else
        f[name] = src_sz
      end
    }
  end

  def add_phrase_pairs pairs
    pairs.each { |i| @additional_phrase_pairs[i] = true }
  end

  def target_ngrams f, s
    ngrams(s, @target_ngrams_n, @target_ngrams_fix) { |ng|
      next if @filter_features && filter(ng)
      name = "NG:"+ng.join("_")
      if @binary
        f[name] = 1.0
      else
        f[name] += ng.size
      end
    }
  end
end

class MosesKbestEntryWithPhraseAlignment < Translation

  def initialize
    super
    @scores[:rr] = -1.0/0
  end

  def get_phrases
    @raw.split(/\|-?\d+\||\|\d+-\d+\|/).map{ |i| i.strip }.reject{ |i| i=='' }
  end

  def _span span
    if span == '-1'
      return [-1]
    else
      return span.split('-').map { |i| i.to_i }
    end
  end

  def get_spans
    @raw.scan(/\|-?\d+\||\|\d+-\d+\|/).map{ |i| i[1..-2] }.map{ |i| _span i }
  end

  def score model
    @scores[:rr] = model.dot(@f)
  end
end

class ConstrainedSearchOracle < MosesKbestEntryWithPhraseAlignment

  def from_s s
    @id = -1
    @raw = s.strip.split(' : ', 2)[1].gsub(/(\[|\])/, '|')
    @s = @raw.gsub(/\s*\|\d+-\d+\||\|-?\d+\|\s*/, ' ').gsub(/\s+/, ' ')
    @scores[:rr] = -1.0/0
  end
end

def structured_update model, hypothesis, oracle, learning_rate
  if hypothesis.s != oracle.s
    model += (oracle.f - hypothesis.f) * learning_rate
    return [model, 1]
  end
  return [model, 0]
end

def ranking_update w, hypothesis, oracle, learning_rate
  if oracle.scores[:rr] <= hypothesis.scores[:rr] \
      && oracle.s != hypothesis.s
    model += (oracle.f - hypothesis.f) * learning_rate
    return [model, 1]
  end
  return [model, 0]
end

def read_additional_phrase_pairs fn
  f = ReadFile.new fn
  add = {}
  while line = f.gets
    id, phrase_pair = line.split ' ', 2
    id = id.to_i-1
    s, t = splitpipe phrase_pair, 3
    phrase_pair = "#{s.strip} ||| #{t.strip}"
    if add.has_key? id
      add[id] << phrase_pair
    else
      add[id] = [phrase_pair]
    end
  end
  return add
end

def usage
  STDERR.write "#{__FILE__} <config file>\n"
  exit 1
end

def main
  usage if ARGV.size != 1
  cfg = read_config ARGV[0]

  sources       = ReadFile.readlines cfg['sources']
  oracles       = ReadFile.readlines cfg['oracles']
  kbest_lists   = read_kbest_lists cfg['kbest_lists'], MosesKbestEntryWithPhraseAlignment
  learning_rate = cfg['learning_rate'].to_f
  learning_rate = 1.0 if !learning_rate
  iterations    = cfg['iterate'].to_i
  output        = WriteFile.new cfg['output']
  output_model  = cfg['output_model']
  silent        = true if cfg['silent']
  verbose       = true if cfg['verbose']
  cheat         = true if cfg['cheat']

  additional_phrase_pairs = nil
  if cfg['additional_phrase_pairs']
    additional_phrase_pairs = read_additional_phrase_pairs cfg['additional_phrase_pairs']
  end

  ff = FeatureFactory.new cfg
  if !silent
    STDERR.write "Running online-reranker with config '#{File.expand_path ARGV[0]}'\n"
    cfg.each_pair { |k,v| STDERR.write "  #{k} = #{v}\n" }
    STDERR.write "\n"
  end

  model = SparseVector.new
  if cfg['init_model']
    model.from_s ReadFile.read cfg['init_model']
  end

  sz = sources.size
  start = Time.now
  iterations.times {
    |t|
  overall_errors = 0
  STDERR.write "Iteration #{t+1} of #{iterations}\n"
  sources.each_with_index { |i,j|
    STDERR.write "  #{j+1}\n" if (j+1)%10==0 && !silent&&!verbose

    ff.add_phrase_pairs(additional_phrase_pairs[j]) if additional_phrase_pairs

    kbest = kbest_lists[j]
    kbest.each { |k|
      k.f = ff.produce k, sources[j]
      k.score model
    }

    hypothesis = kbest[ kbest.map{ |k| k.scores[:rr] }.max_index ]

    if !cheat
      output.write "#{hypothesis.s}\n"
    end

    oracle = ConstrainedSearchOracle.from_s oracles[j]
    oracle.f = ff.produce oracle, sources[j]
    oracle.score model

    err = 0
    case cfg['update']
    when 'structured'
      model, err = structured_update model, hypothesis, oracle, learning_rate
    when 'ranking'
      model, err = ranking_update model, hypothesis, oracle, learning_rate
    else
      STDERR.write "Don't know update method '#{cfg['update']}', exiting.\n"
      exit 1
    end
    overall_errors += err

    if cheat
      kbest.each { |k| k.score model }
      hypothesis = kbest[ kbest.map{ |k| k.scores[:rr] }.max_index ]
      output.write "#{hypothesis.s}\n"
    end

    if verbose
        counts = { 'PP'=>0, 'NG'=>0 }
        model.each_pair { |k,v|
        counts[k.split(':').first] += 1
      }
       STDERR.write "errors=#{overall_errors}; model size=#{model.size} (PP #{counts['PP']}, ng #{counts['NG']})\n" if verbose
    end
  }
  }

  elapsed = Time.now - start
  STDERR.write"#{elapsed.round 2} s, #{(elapsed/Float(sz)).round 2} s per kbest; model size: #{model.size}\n\n" if !silent
  WriteFile.write model.to_s+"\n", output_model if output_model
  output.close
end


main

