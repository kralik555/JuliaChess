using Chess
include("model.jl")
include("board_class.jl")


mutable struct Node
	game::SimpleGame
	args::Dict
	parent::Union{Node, Nothing}
	action_taken::Union{Int, Nothing}
	prior::Float64
	children::Vector{Node}
	visit_count::Integer
	value_sum::Float32
	ucb::Float32

	Node(game, args, parent=nothing, action_taken=nothing, prior=0.0, ucb=0.0) = 
	new(game, args, parent, action_taken, prior, [], 0, 0.0, 0.0)
end


function is_fully_expanded(node::Node)
	return length(node.children) > 0
end


function get_ucb(child::Node, node::Node)
	if child.visit_count == 0
        q_value = Inf
	else
	    q_value = child.value_sum / child.visit_count
    end
	return q_value + node.args["C"] * sqrt(node.visit_count/(child.visit_count + 1)) * child.prior
end

function get_ucb(node::Node)
	return node.ucb
end


function update_ucb(node::Node)
	q_value = node.value_sum / node.visit_count
	node.ucb = q_value + node.args["C"] * sqrt(node.parent.visit_count/(node.visit_count + 1))
end


function select(node::Node)
	#best_child = maximum(get_ucb, node.children)
	best_child = Nothing
	best_ucb = -Inf
	for child in node.children
		ucb = get_ucb(child)
		if ucb > best_ucb
			best_child = child
		end
	end
	return best_child
end

function expand(node::Node, policy)
	for move_idx in eachindex(policy)
		prob = policy[move_idx]
		if prob > 0
			move = int_to_move(move_idx)
			child_state = deepcopy(node.game)
            domove!(child_state, move)
            child = Node(child_state, node.args, node, move_idx, prob, Inf)
			push!(node.children, child)
		end
	end
end


function backpropagate(node::Node, value::Union{Float32, Int64, Float64})
	node.value_sum += value
	node.visit_count += 1
	if node.parent !== nothing
		update_ucb(node)
		backpropagate(node.parent, value)
	end
end


mutable struct MCTS
	game::SimpleGame
	args::Dict
	model::ChessNet
end


function MCTS(game, args, model)
	return MCTS(game, args, model)
end


function search(tree::MCTS)
	root = Node(tree.game, tree.args)
    time0 = time()
    color = sidetomove(tree.game.board)
    searches = 0
    while time() - time0 < tree.args["search_time"]	&& searches < tree.args["num_searches"]
        node = root
        value = 0.0
		while is_fully_expanded(node)
			node = select(node)
		end
		if !(isterminal(node.game))
			policy, value = tree.model.model(board_to_tensor(node.game.board))
			valid_moves = get_valid_moves(node.game.board)
			policy = vec(policy)
			policy = policy .* valid_moves
			policy = policy ./ sum(policy) 
			expand(node, policy)
			value = only(value)
        else
            value = game_result(node.game)
		end
		if color == BLACK
			value = -value
		end
		backpropagate(node, value)
        searches += 1
	end
	action_probs = zeros(Float64, 4096)
	for child in root.children
		action_probs[child.action_taken] = child.visit_count
	end
	action_probs = action_probs ./ sum(action_probs)
    return action_probs, root.value_sum / root.visit_count
end


