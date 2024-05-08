using LinearAlgebra
using MKL
using Chess
using Flux
using JLD2
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
        if game_num >= 50
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
                move = Move(move.from, move.to, QUEEN)
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
                move = Move(move.from, move.to, QUEEN)
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
    while !isterminal(game)
        move_str = readline()
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


if abspath(PROGRAM_FILE) == @__FILE__
    args = Dict{String, Float64}("C" => 1.41, "num_searches" => 500.0, "search_time" => 7.0)
	JLD2.@load "../models/no_info_loss/model_10.jld2" model
    policy, value = model.model(board_to_tensor(startboard()))
    #game_against_computer(model, args)
    model1 = model
    model = nothing
    JLD2.@load "../models/bigger_filter/model_15.jld2" model
    model2 = model
    board = startboard()
    policy, value = model1.model(board_to_tensor(board))
    for i in 1:4096
        if int_to_move(i) in moves(startboard())
            println(int_to_move(i), ": ", policy[i])
        end
    end
    policy, value = model2.model(board_to_tensor(board))
    result = play_game(model1, model2, SimpleGame(startboard()), args)
    println(result)
    result = play_game(model2, model1, SimpleGame(startboard()), args)
    println(result)
end
