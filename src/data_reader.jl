using Chess
using Chess.PGN
using JLD2
using Glob
using SparseArrays
using Serialization
include("board_class.jl")


function most_common_positions(file_path::String, save_path::String, move_depth::Int=8, max_searched::Int=100000)
    position_counter = Dict{String, Int}()
    games_played = 0 
	for game in gamesinfile(file_path)
		games_played += 1
		board = startboard()
       	moves = 0
		is_fine = true
		for move in game.:history
			if move.move == nothing || isterminal(board)
				is_fine = false
				break
			end
			domove!(board, move.move)
			moves += 1
			if moves == move_depth
				break
			end
		end
		if is_fine == true
			position_counter[fen(board)] = get!(position_counter, fen(board), 0) + 1	
			println(fen(board))
		end
		if games_played == max_searched
			break
		end
	end
    # Sort positions by most common and take the top 100
    most_common = sort(collect(position_counter), by=x->x[2], rev=true)[1:100]
    
    # Save to file
    open(save_path, "w") do file
        for (position, _) in most_common
            write(file, position * "\n")
        end
    end
	println(most_common)
end


function get_positions_with_move_distributions(file_path::String, save_path::String, num_games::Int64)
	# dict is fen => array(4096) for move distribution, number of visits, value (wins - losses)
	pos_dict = Dict{String, Tuple{SparseVector{Int, Int}, Int, Int}}()
	games_count = 0
	for game in gamesinfile(file_path)
		games_count += 1
		if games_count == num_games
			break
		end

		if games_count % 1000 == 0
			println(games_count)
		end

		if games_count % 10_000 == 0
			dict_keys = collect(keys(pos_dict))
			for key in dict_keys
				if pos_dict[key][2] == 1
					delete!(pos_dict, key)
				end
			end
			println("After $(games_count):")
			println(length(pos_dict))

		end
		if games_count % 100_000 == 0
			dict_keys = collect(keys(pos_dict))
			for key in dict_keys
				if pos_dict[key][2] < 3
					delete!(pos_dict, key)
				end
			end
		end
		if games_count % 1_000_000 == 0
			dict_keys = collect(keys(pos_dict))
			for key in dict_keys
				if pos_dict[key][2] < 4
					delete!(pos_dict, key)
				end
			end
		end

		board = startboard()

		res = game.headers.:result
		result = res == "1-0" ? 1 : res == "0-1" ? -1 : 0
		
		move_count = 0
		for m in game.:history
			if typeof(m.move) == Nothing || isterminal(game)
				break
			end
			if move_count == 40
				break
			end
			move = tostring(m.move)
			move_int = encode_move(move)
			
			f = fen(board)
			if !(f in keys(pos_dict))
				pos_dict[f] = (spzeros(Int, 4096), 0, 0)
			end
			
			current_data = pos_dict[f]
			new_data = (current_data[1], current_data[2] + 1, current_data[3] + result)
			new_data[1][move_int] += 1
			pos_dict[f] = new_data

			domove!(board, m.move)
			move_count += 1
		end
	end
	
	chunk_size = 50 * 128
	dict_keys = collect(keys(pos_dict))
	num_chunks = ceil(Int, length(dict_keys) / chunk_size)

	for i in 1:num_chunks
		chunk_keys = dict_keys[((i-1)*chunk_size + 1):min(i*chunk_size, end)]
		chunk = Dict(key => pos_dict[key] for key in chunk_keys)
		serialize("$(save_path)chunk_$i.bin", chunk)
	end

end


if abspath(PROGRAM_FILE) == @__FILE__
	pos_dic = get_positions_with_move_distributions("../data/files/data_2016_02.pgn", "../data/move_distros/", 50_001)
	#println(pos_dic["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"])
end
