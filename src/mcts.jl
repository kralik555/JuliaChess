using Chess
include("model.jl")
include("board_class.jl")


mutable struct NodeBoard
    board::Board
    args::Dict
    parent::Union{NodeBoard, Nothing}
	action_taken::Union{Int, Nothing}
	prior::Float64
	children::Vector{NodeBoard}
	visit_count::Integer
	value_sum::Float32

	NodeBoard(board, args, parent=nothing, action_taken=nothing, prior=0.0) = 
	new(board, args, parent, action_taken, prior, [], 0, 0.0)
end


mutable struct Node
	game::SimpleGame
	args::Dict
	parent::Union{Node, Nothing}
	action_taken::Union{Int, Nothing}
	prior::Float64
	children::Vector{Node}
	visit_count::Integer
	value_sum::Float32

	Node(game, args, parent=nothing, action_taken=nothing, prior=0.0) = 
	new(game, args, parent, action_taken, prior, [], 0, 0.0)
end


function is_fully_expanded(node::Union{Node, NodeBoard})
	return length(node.children) > 0
end


function get_ucb(child::Union{Node, NodeBoard}, node::Union{Node, NodeBoard})
	if child.visit_count == 0
		q_value = 0
	else
		q_value = 1 - ((child.value_sum/child.visit_count) + 1) / 2
	end

	return q_value + node.args["C"] * sqrt(node.visit_count/(child.visit_count + 1)) * child.prior
end


# this can be threaded I guess
function select(node::Union{Node, NodeBoard})
	best_child = nothing
	best_ucb = -Inf

	for child in node.children
		ucb = get_ucb(child, node)
		if ucb > best_ucb
			best_child = child
			best_ucb = ucb
		end
	end
	return best_child
end


# this can be theraded
function expand(node::NodeBoard, policy)
	for move_idx in eachindex(policy)
		prob = policy[move_idx]
		if prob > 0
			move = int_to_move(move_idx)
            child_state = domove(node.board, move)
            child = NodeBoard(child_state, node.args, node, move_idx, prob)
			push!(node.children, child)
		end
	end
end


function expand(node::Node, policy)
	for move_idx in eachindex(policy)
		prob = policy[move_idx]
		if prob > 0
			move = int_to_move(move_idx)
            child_state = deepcopy(node.game)
			domove!(child_state, move)
            child = Node(child_state, node.args, node, move_idx, prob)
			push!(node.children, child)
		end
	end
end


function backpropagate(node::Union{Node, NodeBoard}, value::Union{Float32, Int64, Float64})
	node.value_sum = node.value_sum + value
	node.visit_count = node.visit_count + 1
	value = -value
	if node.parent !== nothing
		backpropagate(node.parent, value)
	end
end


mutable struct MCTS
	game::SimpleGame
	args::Dict
	model::ChessNet
end


mutable struct MCTSBoard
    board::Board
    args::Dict
    model::ChessNet
end


function MCTSBoard(board, args, model)
    return MCTSBoard(board, args, model)
end


function MCTS(game, args, model)
	return MCTS(game, args, model)
end


function search(tree::MCTS)
	root = Node(tree.game, tree.args)
	
	for search in 1:tree.args["num_searches"]
		node = root
		
		while is_fully_expanded(node)
			node = select(node)
		end

		if !(isterminal(node.game.board))
			policy, value = model.model(board_to_tensor(node.game.board))
			valid_moves = get_valid_moves(node.game.board)
			policy = vec(policy)
			policy = policy .* valid_moves
			policy = policy ./ sum(policy) 
			expand(node, policy)
        else
            result = game_result(node.game)
            if sidetomove(node.game.board) == Chess.WHITE
				value *= -1
			end
		end
		backpropagate(node, value[1, 1])
	end
	action_probs = zeros(Float64, 4096)
	for child in root.children
		action_probs[child.action_taken] = child.visit_count
	end
	action_probs = action_probs ./ sum(action_probs)
	return action_probs
end


function search(tree::MCTSBoard)
	root = NodeBoard(tree.board, tree.args)
    time_searching = time()
	#for search in 1:tree.args["num_searches"]
    is_first = true
    while time() - time_searching < arguments["search_time"]
        node = root
	    result = 0.0
        is_first = false
		while is_fully_expanded(node)
			node = select(node)
		end

		if !(isterminal(node.board))
            tensors = board_to_tensor(node.board)
			policy, value = model.model(tensors)
            valid_moves = get_valid_moves(node.board)
			policy = vec(policy)
			policy = policy .* valid_moves
			policy = policy ./ sum(policy) 
			expand(node, policy)
        else
            result = game_result(node.board)
            if sidetomove(node.board) == Chess.WHITE
				result *= -1
			end
		end
		backpropagate(node, result)
	end
	action_probs = zeros(Float64, 4096)
	for child in root.children
		action_probs[child.action_taken] = child.visit_count
	end
	action_probs = action_probs ./ sum(action_probs)
    return action_probs, (root.value_sum / root.visit_count)
end
