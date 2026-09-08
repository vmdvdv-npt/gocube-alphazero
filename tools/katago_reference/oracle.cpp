#include "game/board.h"
#include "game/boardhistory.h"
#include "game/rules.h"
#include "external/nlohmann_json/json.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using nlohmann::json;

static constexpr const char* PINNED_COMMIT =
  "f6bc4b19a1686caa2d088b56251e8c11c8be6d51";

struct OracleGame {
  int xSize = 0;
  int ySize = 0;
  Board board;
  BoardHistory history;
  Player nextPlayer = P_BLACK;

  OracleGame(int x, int y, float komi)
    : xSize(x), ySize(y), board(x, y),
      history(board, P_BLACK,
              Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI,
                    false, false, Rules::WHB_ZERO, false, komi),
              0, BoardHistoryModes()) {
    if(komi != 0.5f)
      throw std::runtime_error("KataGo oracle requires komi exactly 0.5");
  }

  Loc point(int x, int y) const {
    if(x < 0 || x >= xSize || y < 0 || y >= ySize)
      throw std::runtime_error("point is outside board");
    return Location::getLoc(x, y, xSize);
  }

  json pointJson(Loc loc) const {
    if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
      return nullptr;
    return json::array({Location::getX(loc, xSize), Location::getY(loc, xSize)});
  }

  void resetPosition(const json& setup) {
    this->board = Board(xSize, ySize);
    if(setup.contains("black"))
      for(const auto& p : setup.at("black"))
        this->board.setStone(point(p.at(0), p.at(1)), C_BLACK);
    if(setup.contains("white"))
      for(const auto& p : setup.at("white"))
        this->board.setStone(point(p.at(0), p.at(1)), C_WHITE);
    // The protocol names capture counts by captured colour, matching
    // KataGo's Board fields.  This is distinct from GoCube's public state,
    // which stores counts by capturing player.
    if(setup.contains("captures")) {
      const auto& captures = setup.at("captures");
      this->board.numBlackCaptures = captures.value("black", 0);
      this->board.numWhiteCaptures = captures.value("white", 0);
    }
    std::string player = setup.value("next_player", "B");
    nextPlayer = (player == "W" || player == "white") ? P_WHITE : P_BLACK;
    int encorePhase = setup.value("encore_phase", 0);
    history.clear(board, nextPlayer, history.rules, encorePhase);
    if(setup.contains("second_cleanup_start_colors")) {
      const auto& colors = setup.at("second_cleanup_start_colors");
      if(!colors.is_array() || colors.size() != static_cast<size_t>(xSize * ySize))
        throw std::runtime_error("second_cleanup_start_colors must contain one value per point");
      for(int y = 0; y < ySize; y++)
        for(int x = 0; x < xSize; x++)
          history.secondEncoreStartColors[point(x, y)] = colors.at(y * xSize + x).get<int>();
    }
  }

  json snapshot() const {
    json result;
    Color area[Board::MAX_ARR_SIZE];
    bool nonPassAliveStones = false;
    bool safeBigTerritories = false;
    bool unsafeBigTerritories = false;
    board.calculateArea(
      area,
      nonPassAliveStones,
      safeBigTerritories,
      unsafeBigTerritories,
      false
    );
    bool allPointsPassAlive = true;
    for(int y = 0; y < ySize && allPointsPassAlive; y++)
      for(int x = 0; x < xSize; x++)
        if(area[point(x, y)] == C_EMPTY) {
          allPointsPassAlive = false;
          break;
        }
    result["ok"] = true;
    result["katago_commit"] = PINNED_COMMIT;
    result["board"] = json::array();
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++)
        result["board"].push_back((int)board.colors[point(x, y)]);
    result["next_player"] = nextPlayer == P_BLACK ? "B" : "W";
    result["legal_mask"] = json::array();
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++)
        result["legal_mask"].push_back(
          (!history.isGameFinished && !history.isNoResult && history.isLegal(board, point(x, y), nextPlayer)) ? 1 : 0
        );
    result["legal_mask"].push_back(
      (!history.isGameFinished && !history.isNoResult && history.isLegal(board, Board::PASS_LOC, nextPlayer)) ? 1 : 0
    );

    if(history.isNoResult)
      result["phase"] = "NO_RESULT";
    else if(history.isGameFinished)
      result["phase"] = "SCORED";
    else if(history.encorePhase == 0)
      result["phase"] = "MAIN";
    else if(history.encorePhase == 1)
      result["phase"] = "CLEANUP_1";
    else
      result["phase"] = "CLEANUP_2";

    result["simple_ko"] = pointJson(board.ko_loc);
    result["ko_recap_blocked"] = json::array();
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++)
        if(history.koRecapBlocked[point(x, y)])
          result["ko_recap_blocked"].push_back(json::array({x, y}));
    result["is_game_finished"] = history.isGameFinished;
    result["is_no_result"] = history.isNoResult;
    if(history.winner == P_BLACK)
      result["winner"] = "black";
    else if(history.winner == P_WHITE)
      result["winner"] = "white";
    else if(history.isGameFinished)
      result["winner"] = "draw";
    else
      result["winner"] = nullptr;
    result["final_score"] = history.isScored
      ? json(history.finalWhiteMinusBlackScore) : json(nullptr);
    result["captures"] = {
      {"black", board.numBlackCaptures},
      {"white", board.numWhiteCaptures}
    };
    result["all_points_pass_alive"] = allPointsPassAlive;
    result["encore_phase"] = history.encorePhase;
    Color formalArea[Board::MAX_ARR_SIZE];
    history.getAreaNow(board, formalArea);
    result["formal_area"] = json::array();
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++)
        result["formal_area"].push_back((int)formalArea[point(x, y)]);
    result["white_bonus_score"] = history.whiteBonusScore;
    result["second_cleanup_start_colors"] = json::array();
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++)
        result["second_cleanup_start_colors"].push_back(
          (int)history.secondEncoreStartColors[point(x, y)]
        );
    return result;
  }

  json play(const json& move) {
    Loc loc = Board::PASS_LOC;
    if(move.is_array())
      loc = point(move.at(0), move.at(1));
    else if(move.is_string() && move.get<std::string>() != "pass")
      throw std::runtime_error("move must be [x,y] or pass");

    if(!history.isLegal(board, loc, nextPlayer))
      return {{"ok", false}, {"error", "illegal"}, {"snapshot", snapshot()}};
    Player played = nextPlayer;
    history.makeBoardMoveAssumeLegal(board, loc, played, nullptr);
    nextPlayer = getOpp(played);
    return snapshot();
  }
};

int main() {
  try {
    Board::initHash();
    std::string line;
    std::unique_ptr<OracleGame> game;
    while(std::getline(std::cin, line)) {
      if(line.empty())
        continue;
      try {
        json command = json::parse(line);
        std::string op = command.at("op").get<std::string>();
        if(op == "new") {
          int x = command.at("x_size").get<int>();
          int y = command.at("y_size").get<int>();
          float komi = command.at("komi").get<float>();
          game = std::make_unique<OracleGame>(x, y, komi);
          std::cout << game->snapshot().dump() << std::endl;
        } else if(op == "setup") {
          if(!game)
            throw std::runtime_error("setup requires new first");
          game->resetPosition(command);
          std::cout << game->snapshot().dump() << std::endl;
        } else if(op == "play") {
          if(!game)
            throw std::runtime_error("play requires new first");
          std::cout << game->play(command.at("move")).dump() << std::endl;
        } else if(op == "snapshot") {
          if(!game)
            throw std::runtime_error("snapshot requires new first");
          std::cout << game->snapshot().dump() << std::endl;
        } else {
          throw std::runtime_error("unknown oracle operation");
        }
      } catch(const std::exception& exc) {
        std::cout << json({
          {"ok", false},
          {"error", exc.what()},
          {"katago_commit", PINNED_COMMIT}
        }).dump() << std::endl;
      }
    }
  } catch(const std::exception& exc) {
    std::cerr << exc.what() << std::endl;
    return 2;
  }
  return 0;
}
