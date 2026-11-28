// Skeleton expansion of composite kernels (KernelImpl::expand / Options::expand_kernel_graphs).

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using carbon::detail::KernelGraph;
using carbon::detail::KernelImpl;

using LaunchLog = std::vector<std::string>;

class LoggingKernel final : public KernelImpl {
public:
  LoggingKernel(std::string name,
                std::vector<carbon::AccessToken> tokens,
                std::shared_ptr<LaunchLog> log,
                bool communicates = false,
                std::function<std::optional<KernelGraph>()> graph = {})
      : KernelImpl(std::move(name)),
        tokens_(std::move(tokens)),
        log_(std::move(log)),
        communicates_(communicates),
        graph_(std::move(graph)) {}

  auto
  launch(const carbon::Stream& /*stream*/,
         carbon::DataView /*view*/) -> void override {
    log_->push_back(name_);
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  kind() const noexcept -> carbon::KernelKind override {
    return carbon::KernelKind::Halo;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<carbon::AccessToken>& override {
    return tokens_;
  }

  [[nodiscard]] auto
  communicates() const noexcept -> bool override {
    return communicates_;
  }

  [[nodiscard]] auto
  expand(const std::shared_ptr<KernelImpl>& /*self*/) const -> std::optional<KernelGraph> override {
    if (!graph_) {
      return std::nullopt;
    }
    return graph_();
  }

private:
  std::vector<carbon::AccessToken> tokens_;
  std::shared_ptr<LaunchLog> log_;
  bool communicates_;
  std::function<std::optional<KernelGraph>()> graph_;
};

[[nodiscard]] auto
build_jacobi(carbon::Dense& dense,
             carbon::Dense::Field<float,
                                  1>& fin,
             carbon::Dense::Field<float,
                                  1>& fout,
             std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
          const float self = in.read(s, idx, 0);
          if (!xp.valid || !xm.valid) {
            out.write(s, idx, 0, self);
            return;
          }
          out.write(s, idx, 0, 0.5F * (xp.value + xm.value));
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
line_index_of(const std::string& text,
              const std::string& needle) -> std::size_t {
  const std::size_t pos = text.find(needle);
  if (pos == std::string::npos) {
    return static_cast<std::size_t>(-1);
  }
  return static_cast<std::size_t>(
      std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(pos), '\n'));
}

[[nodiscard]] auto
index_in(const LaunchLog& log,
         const std::string& name) -> std::size_t {
  const auto it = std::find(log.begin(), log.end(), name);
  return it == log.end() ? static_cast<std::size_t>(-1)
                         : static_cast<std::size_t>(it - log.begin());
}

struct Fixture {
  carbon::System& sys;
  carbon::Dense& dense;
  carbon::Dense::Field<float, 1>& u;
  carbon::Dense::Field<float, 1>& v;
  std::vector<carbon::AccessToken> halo_tokens;
};

// A composite named "comp" over the halo tokens of u; its children carry the same tokens.
[[nodiscard]] auto
make_composite(const Fixture& fx,
               const std::shared_ptr<LaunchLog>& log,
               std::function<std::optional<KernelGraph>()> graph,
               std::string name = "comp") -> carbon::Kernel {
  return carbon::Kernel::from_impl(std::make_shared<LoggingKernel>(
      std::move(name), fx.halo_tokens, log, /*communicates=*/true, std::move(graph)));
}

[[nodiscard]] auto
child(const Fixture& fx,
      const std::shared_ptr<LaunchLog>& log,
      std::string name,
      bool communicates = false) -> std::shared_ptr<KernelImpl> {
  return std::make_shared<LoggingKernel>(std::move(name), fx.halo_tokens, log, communicates);
}

[[nodiscard]] auto
sizes_consistent(const carbon::Skeleton& sk) -> bool {
  const std::size_t n = sk.effective_kernels().size();
  return sk.effective_to_user().size() == n && sk.effective_parent().size() == n &&
         sk.kernel_streams().size() == n && sk.edges().size() == n && sk.min_edges().size() == n;
}

#define CHECK(cond, ...)                                            \
  do {                                                              \
    if (!(cond)) {                                                  \
      CARBON_LOG_ERROR("test_skeleton_kernel_graph: " __VA_ARGS__); \
      return false;                                                 \
    }                                                               \
  } while (0)

auto
test_chain(const Fixture& fx,
           carbon::OverlapMode mode) -> bool {
  auto log = std::make_shared<LaunchLog>();
  auto graph = [&fx, log]() -> std::optional<KernelGraph> {
    KernelGraph g;
    const size_t a = g.add(child(fx, log, "comp/a"));
    const size_t b = g.add(child(fx, log, "comp/b", /*communicates=*/true));
    const size_t c = g.add(child(fx, log, "comp/c"));
    g.edge(KernelGraph::kHead, a);
    g.edge(a, b);
    g.edge(b, c);
    g.edge(c, KernelGraph::kTail);
    return g;
  };
  carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{mode});
  sk.add(make_composite(fx, log, graph));
  sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
  sk.compile();

  CHECK(sizes_consistent(sk), "chain: per-kernel vectors not resized");
  const auto& eff = sk.effective_kernels();
  CHECK(eff.size() == 5U, "chain: expected 2 + 3 effective kernels, got {}", eff.size());
  for (size_t k = 2; k < 5; ++k) {
    CHECK(sk.effective_parent()[k] == 0U, "chain: child {} has parent {}", k,
          sk.effective_parent()[k]);
    CHECK(sk.effective_to_user()[k] == carbon::Skeleton::kNoKernel, "chain: child {} maps to user",
          k);
  }
  const auto& last_child_succs = sk.min_edges()[4];
  CHECK(std::find(last_child_succs.begin(), last_child_succs.end(), 1U) != last_child_succs.end(),
        "chain: last child does not feed the stencil kernel in min_edges");

  const std::string order = sk.export_launch_order();
  CHECK(order.find("launch  comp ") == std::string::npos &&
            order.find("launch  comp\n") == std::string::npos,
        "chain: composite itself is launched\n{}", order);
  const size_t c_line = line_index_of(order, "comp/c");
  const size_t step_line = mode == carbon::OverlapMode::None
                               ? line_index_of(order, "launch  step")
                               : line_index_of(order, "[compute/boundary]");
  CHECK(c_line != static_cast<size_t>(-1) && step_line != static_cast<size_t>(-1) &&
            c_line < step_line,
        "chain: stencil kernel must follow the last child\n{}", order);

  sk.run();
  sk.wait();
  const LaunchLog expected{"comp/a", "comp/b", "comp/c"};
  CHECK(*log == expected, "chain: launch log mismatch ({} entries)", log->size());

  const auto msgs = sk.halo_messages();
  CHECK(msgs.size() == 1U, "chain: halo_messages should list only the communicating child, got {}",
        msgs.size());
  return true;
}

auto
test_diamond_and_dangling(const Fixture& fx) -> bool {
  auto log = std::make_shared<LaunchLog>();
  auto graph = [&fx, log]() -> std::optional<KernelGraph> {
    KernelGraph g;
    const size_t a = g.add(child(fx, log, "comp/a"));
    const size_t b = g.add(child(fx, log, "comp/b"));
    const size_t c = g.add(child(fx, log, "comp/c"));
    const size_t d = g.add(child(fx, log, "comp/d"));
    g.add(child(fx, log, "comp/lonely"));
    g.edge(KernelGraph::kHead, a);
    g.edge(a, b);
    g.edge(a, c);
    g.edge(b, d);
    g.edge(c, d);
    g.edge(d, KernelGraph::kTail);
    return g;
  };
  carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
  sk.add(make_composite(fx, log, graph));
  sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
  sk.compile();
  CHECK(sizes_consistent(sk), "diamond: per-kernel vectors not resized");
  const std::string order = sk.export_launch_order();
  const size_t step_line = line_index_of(order, "launch  step");
  for (const char* name : {"comp/a", "comp/b", "comp/c", "comp/d", "comp/lonely"}) {
    const size_t line = line_index_of(order, name);
    CHECK(line != static_cast<size_t>(-1) && line < step_line,
          "diamond: child {} must launch before the stencil kernel\n{}", name, order);
  }
  sk.run();
  sk.wait();
  CHECK(log->size() == 5U, "diamond: expected 5 launches, got {}", log->size());
  const size_t ia = index_in(*log, "comp/a");
  const size_t ib = index_in(*log, "comp/b");
  const size_t ic = index_in(*log, "comp/c");
  const size_t id = index_in(*log, "comp/d");
  CHECK(ia < ib && ia < ic && ib < id && ic < id, "diamond: launch order violates the edges");
  return true;
}

auto
test_empty_graph(const Fixture& fx) -> bool {
  auto log = std::make_shared<LaunchLog>();
  auto graph = []() -> std::optional<KernelGraph> {
    return KernelGraph{};
  };
  carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
  sk.add(make_composite(fx, log, graph));
  sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
  sk.compile();
  CHECK(sizes_consistent(sk), "empty: per-kernel vectors not resized");
  CHECK(sk.effective_kernels().size() == 2U, "empty: no child expected");
  const std::string order = sk.export_launch_order();
  CHECK(order.find("launch  step") != std::string::npos, "empty: stencil kernel missing\n{}",
        order);
  sk.run();
  sk.wait();
  CHECK(log->empty(), "empty: nothing should be launched by the composite");
  return true;
}

auto
test_nested(const Fixture& fx) -> bool {
  auto log = std::make_shared<LaunchLog>();
  auto inner = [&fx, log]() -> std::optional<KernelGraph> {
    KernelGraph g;
    const size_t x = g.add(child(fx, log, "comp/mid/x"));
    const size_t y = g.add(child(fx, log, "comp/mid/y"));
    g.edge(x, y);
    return g;
  };
  auto outer = [&fx, log, inner]() -> std::optional<KernelGraph> {
    KernelGraph g;
    const size_t a = g.add(child(fx, log, "comp/a"));
    const size_t mid =
        g.add(std::make_shared<LoggingKernel>("comp/mid", fx.halo_tokens, log, false, inner));
    const size_t c = g.add(child(fx, log, "comp/c"));
    g.edge(a, mid);
    g.edge(mid, c);
    return g;
  };
  carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
  sk.add(make_composite(fx, log, outer));
  sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
  sk.compile();
  CHECK(sizes_consistent(sk), "nested: per-kernel vectors not resized");
  CHECK(sk.effective_kernels().size() == 7U, "nested: expected 2 + 3 + 2 effective kernels, got {}",
        sk.effective_kernels().size());
  // comp/mid is effective kernel 3; its children are 5 and 6.
  CHECK(sk.effective_parent()[5] == 3U && sk.effective_parent()[6] == 3U,
        "nested: grandchildren should point at the inner composite");
  sk.run();
  sk.wait();
  const LaunchLog expected{"comp/a", "comp/mid/x", "comp/mid/y", "comp/c"};
  CHECK(*log == expected, "nested: launch log mismatch ({} entries)", log->size());
  return true;
}

auto
test_rejects(const Fixture& fx) -> bool {
  auto log = std::make_shared<LaunchLog>();
  {
    auto cyclic = [&fx, log]() -> std::optional<KernelGraph> {
      KernelGraph g;
      const size_t a = g.add(child(fx, log, "comp/a"));
      const size_t b = g.add(child(fx, log, "comp/b"));
      g.edge(KernelGraph::kHead, a);
      g.edge(a, b);
      g.edge(b, a);
      g.edge(b, KernelGraph::kTail);
      return g;
    };
    carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk.add(make_composite(fx, log, cyclic));
    sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
    bool threw = false;
    try {
      sk.compile();
    } catch (const std::logic_error&) {
      threw = true;
    }
    CHECK(threw, "rejects: a cyclic kernel graph must throw");
  }
  {
    // The stencil kernel writes v, which the composite's halo tokens do not cover.
    const carbon::Kernel jacobi = build_jacobi(fx.dense, fx.u, fx.v, "uncovered");
    auto uncovered = [&fx, log, jacobi]() -> std::optional<KernelGraph> {
      KernelGraph g;
      g.add(std::make_shared<LoggingKernel>("comp/bad", jacobi.impl()->tokens(), log));
      return g;
    };
    carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk.add(make_composite(fx, log, uncovered));
    sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
    bool threw = false;
    try {
      sk.compile();
    } catch (const std::logic_error&) {
      threw = true;
    }
    CHECK(threw, "rejects: a child accessing beyond the composite's tokens must throw");
  }
  CHECK(log->empty(), "rejects: nothing should have run");
  return true;
}

auto
test_opt_out(const Fixture& fx) -> bool {
  auto log = std::make_shared<LaunchLog>();
  auto graph = [&fx, log]() -> std::optional<KernelGraph> {
    KernelGraph g;
    g.add(child(fx, log, "comp/a"));
    return g;
  };
  carbon::Skeleton::Options opts{carbon::OverlapMode::None};
  opts.expand_kernel_graphs = false;
  carbon::Skeleton sk(fx.sys, opts);
  sk.add(make_composite(fx, log, graph));
  sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step"));
  sk.compile();
  CHECK(sizes_consistent(sk), "opt-out: per-kernel vectors inconsistent");
  CHECK(sk.effective_kernels().size() == 2U, "opt-out: composite must not be expanded");
  sk.run();
  sk.wait();
  const LaunchLog expected{"comp"};
  CHECK(*log == expected, "opt-out: the composite itself should run once");
  return true;
}

auto
test_extended(const Fixture& fx) -> bool {
  // v = J(u); u = J(v): the second step gets an auto halo on v, the first the composite on u.
  auto log = std::make_shared<LaunchLog>();
  auto graph = [&fx, log]() -> std::optional<KernelGraph> {
    KernelGraph g;
    const size_t a = g.add(child(fx, log, "comp/a"));
    const size_t b = g.add(child(fx, log, "comp/b", /*communicates=*/true));
    g.edge(a, b);
    return g;
  };
  carbon::Skeleton sk(fx.sys, carbon::Skeleton::Options{carbon::OverlapMode::Extended});
  sk.add(make_composite(fx, log, graph));
  sk.add(build_jacobi(fx.dense, fx.u, fx.v, "step_uv"));
  sk.add(build_jacobi(fx.dense, fx.v, fx.u, "step_vu"));
  sk.compile();
  CHECK(sizes_consistent(sk), "extended: per-kernel vectors not resized");
  for (int iter = 0; iter < 2; ++iter) {
    sk.run();
  }
  sk.wait();
  const LaunchLog expected{"comp/a", "comp/b", "comp/a", "comp/b"};
  CHECK(*log == expected, "extended: launch log mismatch ({} entries)", log->size());
  return true;
}

#undef CHECK

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_kernel_graph: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{1, 0, 0}, {-1, 0, 0}},
                          /*remove_center=*/true);
  carbon::Dense dense(sys, {8, 6, 64}, stencil);
  auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v");
  const carbon::Kernel halo_u = u.make_halo_kernel("halo_u");
  Fixture fx{sys, dense, u, v, halo_u.impl()->tokens()};

  bool ok = true;
  ok = test_chain(fx, carbon::OverlapMode::None) && ok;
  ok = test_chain(fx, carbon::OverlapMode::Standard) && ok;
  ok = test_diamond_and_dangling(fx) && ok;
  ok = test_empty_graph(fx) && ok;
  ok = test_nested(fx) && ok;
  ok = test_rejects(fx) && ok;
  ok = test_opt_out(fx) && ok;
  ok = test_extended(fx) && ok;
  if (!ok) {
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_skeleton_kernel_graph: OK");
  return EXIT_SUCCESS;
}
