using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")

function model_move(model, board)
    # Convert the board to a tensor representation suitable for your model
	tensor = board_to_tensor(board)
	moves, value = model.model(tensor)
	valid_moves = get_valid_moves(board)                           
    moves = vec(moves)                                                
    moves = moves .* valid_moves                                      
    moves = moves ./ sum(moves)                                      

    # Choose the move with the highest probability
    best_move_index = argmax(moves[:])
	return best_move_index
end

function tree_move(model, board, args)
	tree = MCTS(board, args, model)
	move_probs = search(tree)
	move = argmax(move_probs)
	return move
end
