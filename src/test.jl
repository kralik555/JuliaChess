using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")
include("moves.jl")
include("data_reader.jl")

function test_models(model1::ChessNet, model2::ChessNet, positions_file::String, args::Dict{String, Union{Int, Float64}})
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

function play_game(model1::ChessNet, model2::ChessNet, game::SimpleGame, args::Dict{String, Union{Int64, Float64}})
    while !isterminal(game)
        move = tree_move(model1, game, args)
        move = int_to_move(Int(only(move)))
        println(tostring(move))
        domove!(game, move)
        if isterminal(game)
            break
        end
        move = tree_move(model2, game, args)
        move = int_to_move(Int(only(move)))
        println(tostring(move))
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
	JLD2.@load "../models/self_play_models/model_2.jld2" model
    model1 = model
    model = nothing
    JLD2.@load "../models/supervised_model_1.jld2" model
    model2 = model
    model = nothing
    args = Dict{String, Union{Int, Float64}}("C" => 2, "num_searches" => 100, "search_time" => 1.0)
    test_models(model1, model2, "../data/common_games.txt", args)
end
