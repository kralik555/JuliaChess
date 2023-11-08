using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")

function model_move(board, model)
    # Convert the board to a tensor representation suitable for your model
	tensor = board_to_tensor(board)
	moves, value = model.model(tensor)
	valid_moves = get_valid_moves(board)                           
    moves = vec(moves)                                                
    moves = moves .* valid_moves                                      
    moves = moves ./ sum(moves)                                      

    # Choose the move with the highest probability
    best_move_index = argmax(moves[:])
	println(moves)
	return best_move_index
end

function tree_move(model, board, args)
	tree = MCTS(board, args, model)
	move_probs = search(tree)
	move = argmax(move_probs)
	return move
end

function play_game(model)
    board = startboard()

    while !isterminal(board)
        #move = model_move(board, model)
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
end



if abspath(PROGRAM_FILE) == @__FILE__
	JLD2.@load "models/model_high_rating.jld2" model
	print_model_details(model)
	args = Dict("C" => 2, "num_searches" => 100)
	play_game(model)
end
