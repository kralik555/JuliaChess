using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")

function model_move(model::ChessNet, board::Board)
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

function tree_move(model::ChessNet, game::SimpleGame, args::Dict{String, Float64})
	tree = MCTS(game, args, model)
	move_probs = search(tree)
	move = argmax(move_probs)
	return move
end

function tree_move_with_distro(model::ChessNet, board::Board, args::Dict{String, Union{Float64, Int64}})
    tree = MCTSBoard(board, args, model)
    move_probs, value = search(tree)
    move = argmax(move_probs)
    return (move_probs, move, value)
end

function model_move_with_distros(model::ChessNet, board::Board)
    probs, _ = model.model(board_to_tensor(board))
    valid_moves = get_valid_moves(board)
    probs = vec(probs)
    probs = probs .* valid_moves
    probs = probs ./ sum(probs)
    return (probs, argmax(probs[:]))
end
