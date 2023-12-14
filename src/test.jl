using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")
include("moves.jl")

function test_models(model1::ChessNet, model2::ChessNet, positions_file::String)
	model1_wins = 0
	model2_wins = 0
	draws = 0

	# for each position in positions_file play a game
	# store result

	return (model1_wins, draws, model2_wins)
end

function play_game(model::ChessNet, starting_position::String, args::Dict{String, Int64})
    if starting_position == ""
		board = startboard()
	else 
		board = fromfen(starting_position)
	end

    while !isterminal(board)
		#move = model_move(model, board)
		move = tree_move(model, board, args)
		move = int_to_move(Int(only(move)))
		println(move)
        domove!(board, move)
    end

    result = 0
    if isdraw(board)
	    nothing
    elseif sidetomove(board) == WHITE
	    result = -1
    else
	    retuls = 1
    end
    println("Result of the game: ", result)
	return result
end



if abspath(PROGRAM_FILE) == @__FILE__
	JLD2.@load "../models/supervised_model.jld2" model
	args = Dict("C" => 2, "num_searches" => 100)
	play_game(model, "", args)
end
