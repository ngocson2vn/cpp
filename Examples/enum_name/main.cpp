#include <cstdio>
#include <iostream>
#include <array>

enum class ExprKind {
  Expr_VarDecl,
  Expr_Return,
  Expr_Num,
  Expr_Literal,
  Expr_Var,
  Expr_AssignOp,
  Expr_BinOp,

  // Host ops
  Expr_Random,
  Expr_Call,
  Expr_Print,

  // Device ops
  Expr_Add,
  Expr_Mul,
  Expr_Max,
  Expr_Dot,

  Count
};

template <ExprKind V>
constexpr std::string_view kindName() {
  std::string_view funcName = __PRETTY_FUNCTION__;
  auto eqPos = funcName.find("=");
  auto rpart = funcName.substr(eqPos + 2);
  return rpart.substr(0, rpart.size() - 1);
}

template <std::size_t... I>
constexpr auto makeKindNames(std::index_sequence<I...>) {
  return std::array<std::string_view, sizeof...(I)>{
      kindName<static_cast<ExprKind>(I)>()...};
}

inline constexpr auto kKindNames = makeKindNames(
    std::make_index_sequence<static_cast<std::size_t>(ExprKind::Count)>{});

std::string_view getKindName(ExprKind kind) {
  auto kindIdx = static_cast<std::size_t>(kind);
  return kKindNames[kindIdx];
}

int main(int argc, char** argv) {
  std::cout << getKindName(ExprKind::Expr_Add) << std::endl;

  return 0;
}
