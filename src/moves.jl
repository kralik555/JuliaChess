using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")

function model_move(model::ChessNet, board::Board)
	tensor = board_to_tensor(board)
	moves, value = model.model(tensor)
	valid_moves = get_valid_moves(board)                           
    moves = vec(moves)                                                
    moves = moves .* valid_moves                                      
    moves = moves ./ sum(moves)                                      

    best_move_index = argmax(moves[:])
    move = int_to_move(best_move_index)
	return move
end

function tree_move(model::ChessNet, game::SimpleGame, args::Dict{String, Union{Float64, Int64}})
	tree = MCTS(game, args, model)
	move_probs, value = search(tree)
	move = argmax(move_probs)
	return move_probs, move, value
end


function model_move_with_distros(model::ChessNet, board::Board)
    probs, _ = model.model(board_to_tensor(board))
    valid_moves = get_valid_moves(board)
    probs = vec(probs)
    probs = probs .* valid_moves
    probs = probs ./ sum(probs)
    return (probs, argmax(probs[:]))
end
