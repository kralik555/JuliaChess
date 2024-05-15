using LinearAlgebra
using MKL
using Chess
using Flux
using JLD2
using Chess.UCI
include("board_class.jl")
include("model.jl")
include("mcts.jl")
include("moves.jl")
include("data_reader.jl")

function test_models(model1::ChessNet, model2::ChessNet, positions_file::String, args::Dict{String, Float64})
	model1_wins = 0
	model2_wins = 0
	draws = 0
    
    positions = load_most_common(positions_file)

    sth, _ = model1.model(board_to_tensor(startboard()))
    sth, _ = model2.model(board_to_tensor(startboard()))
    game_num = 0
    for position in positions
        game_num += 1
        if game_num >= 10
            break
        end
        println("Game ", game_num)
        game = SimpleGame(fromfen(position))
        result1 = play_game(model1, model2, game, args)
        println(result1)
        game = SimpleGame(fromfen(position))
        result2 = play_game(model2, model1, game, args)
        println(result2)
        result = result1 - result2
        println("Result for this round: ", result1 - result2)
        if result > 0
            model1_wins += 1
        elseif result == 0
            draws += 1
        else
            model2_wins += 1
        end
        println("Model 1 wins: ", model1_wins)
        println("Model 2 wins: ", model2_wins)
    end
    
	return (model1_wins, draws, model2_wins)
end

function play_game(model1::ChessNet, model2::ChessNet, game::SimpleGame, args::Dict{String, Float64})
    while !isterminal(game)
        probs, move, value = tree_move(model1, game, args)
        move = int_to_move(Int(only(move)))
        println(tostring(move))
        if ptype(pieceon(game.board, from(move))) == PAWN
            if Chess.rank(to(move)) == RANK_8 || Chess.rank(to(move)) == RANK_1
                move = Move(from(move), to(move), QUEEN)
            end
        end
        domove!(game, move)
        if isterminal(game)
            break
        end
        probs, move, value = tree_move(model2, game, args)
        move = int_to_move(Int(only(move)))
        println(tostring(move))
        if ptype(pieceon(game.board, from(move))) == PAWN
            if Chess.rank(to(move)) == RANK_8 || Chess.rank(to(move)) == RANK_1
                move = Move(from(move), to(move), QUEEN)
            end
        end
        domove!(game, move)
    end
    println(gametopgn(game))
    result = game_result(game)

    return result
end

function play_self_game(model::ChessNet, starting_position::String, args::Dict{String, Float64})
    if starting_position == ""
		board = startboard()
	else 
		board = fromfen(starting_position)
	end
    
    game = SimpleGame(board)

    while !(isterminal(game))
		policy, move, value = tree_move(model, game, args)
		move = int_to_move(Int(only(move)))
		println(move)
        domove!(game, move)
    end
	
    result = game_result(game)
    println("Result of the game: ", result)
    println(lichessurl(game.board))
	return result
end

function game_against_computer(model::ChessNet, args)
    board = startboard()
    game = SimpleGame(board)
    println("Enter moves in format e2e4 - move from e2 to e4. For promotion, type b7b8q - change q to any piece you want -- q, r, n, b")
    while !isterminal(game)
        move_str = readline("Enter a move: ")
        move = movefromstring(move_str)
        domove!(game, move)
        if !(isterminal(game))
            probs, move, value = tree_move(model, game, args)
            move = int_to_move(Int(only(move)))
            domove!(game, move)
            println(move)
        end
    end

    result = game_result(game)
    println(result)
end

function stockfish_game(model::ChessNet, stockfish_path::String, game::SimpleGame, args)
    engine = runengine(stockfish_path)
    setboard(engine, game.board)

    while !isterminal(game)
        probs, move, value = tree_move(model, game, args)
        move = int_to_move(Int(only(move)))
        domove!(game, move)
        setboard(engine, game.board)
        if !(isterminal(game))
            move = Chess.UCI.search(engine, "go depth 10")
            domove!(game, move.bestmove)
        end
    end
    result = game_result(game)
    println(lichessurl(game.board))
    return result
end

function stockfish_game(stockfish_path::String, model::ChessNet, game::SimpleGame, args)
    engine = runengine(stockfish_path)
    setboard(engine, game.board)
    while !isterminal(game)
        move = Chess.UCI.search(engine, "go depth 10")
        domove!(game, move.bestmove)
        if !(isterminal(game))
            probs, move, value = tree_move(model, game, args)
            move = int_to_move(Int(only(move)))
            domove!(game, move)
            setboard(engine, game.board)
        end
    end
    result = game_result(game)
    println(lichessurl(game.board))
    return result
end

function test_against_stockfish(model::ChessNet, args, stockfish_path::String, positions_path::String)
    model_wins = 0
    stockfish_wins = 0
    draws = 0
    positions = load_most_common(positions_path)

    sth, _ = model.model(board_to_tensor(startboard()))
    for position in positions[1:50]
        game = SimpleGame(fromfen(position))
        result1 = stockfish_game(model, stockfish_path, game, args)
        if result1 > 0
            model_wins += 1
        elseif result1 == 0
            draws += 1
        else
            stockfish_wins += 1
        end
        result2 = stockfish_game(stockfish_path, model, game, args)
        if result2 > 0
            stockfish_wins += 1
        elseif result2 == 0
            draws += 1
        else
            model_wins += 1
        end
    end
    return model_wins, stockfish_wins, draws
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = Dict{String, Float64}("C" => 0.7, "num_searches" => 500.0, "search_time" => 7.0)
    model = ChessNet()
    JLD2.@load "../models/final_model.jld2" model
    policy, value = model.model(board_to_tensor(startboard()))
        
    game_against_computer(model, args)
end
