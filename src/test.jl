using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")
include("moves.jl")
include("data_reader.jl")

function test_models(model1::ChessNet, model2::ChessNet, positions_file::String, args::Dict{String, Int})
	model1_wins = 0
	model2_wins = 0
	draws = 0
    
    positions = load_most_common(positions_file)

    for position in positions
        game = SimpleGame(fromfen(position))
        result1 = play_game(model1, model2, game, args)
        result2 = play_game(model2, model1, game, args)
        result = result1 - result2
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

function play_game(model1::ChessNet, model2::ChessNet, game::SimpleGame, args::Dict{String, Int})
    while !(game_over(game)[1])
        move = tree_move(model, game, args)
        move = int_to_move(Int(only(move)))
        println(move)
        domove!(game, move)
    end

    res = game.headers.:result
    result = res == "1-0" ? 1 : res == "0-1" ? -1 : 0
    return result
end

function play_self_game(model::ChessNet, starting_position::String, args::Dict{String, Int64})
    if starting_position == ""
		board = startboard()
	else 
		board = fromfen(starting_position)
	end
    
    game = SimpleGame(board)

    while !(isterminal(game))
		move = tree_move(model, game, args)
		move = int_to_move(Int(only(move)))
		println(move)
        domove!(game, move)
    end
	
    result = 0
    res = game.headers.:result
    result = res == "1-0" ? 1 : res == "0-1" ? -1 : 0
    println("Result of the game: ", result)
	return result
end



if abspath(PROGRAM_FILE) == @__FILE__
	JLD2.@load "../models/supervised_model.jld2" model
	args = Dict("C" => 2, "num_searches" => 100)
	play_self_game(model, "", args)
end
