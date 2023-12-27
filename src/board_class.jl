using Chess
using Flux
import Chess


function create_input_tensors(board::Board)
    tensor = zeros(UInt8, 18, 8, 8)
    
    # Mapping of pieces to tensor planes
    piece_map = Dict(
        PIECE_WK => 1, PIECE_WQ => 2, PIECE_WR => 3, PIECE_WB => 4, PIECE_WN => 5, PIECE_WP => 6,
        PIECE_BK => 7, PIECE_BQ => 8, PIECE_BR => 9, PIECE_BB => 10, PIECE_BN => 11, PIECE_BP => 12
    )
    
    # Fill the tensor for pieces
    for tile = 1:64
	    piece = pieceon(board, Square(tile))
            if piece !== EMPTY
		    tensor[piece_map[piece], div(tile - 1, 8) + 1, (tile - 1) % 8 + 1] = 1
            end
        end 
    
    # Castling rights
    tensor[13, :, :] .= cancastlekingside(board, WHITE) ? 1 : 0
    tensor[14, :, :] .= cancastlequeenside(board, WHITE) ? 1 : 0
    tensor[15, :, :] .= cancastlekingside(board, BLACK) ? 1 : 0
    tensor[16, :, :] .= cancastlequeenside(board, BLACK) ? 1 : 0
    
    # Player's turn
    tensor[17, :, :] .= sidetomove(board) == WHITE ? 1 : 0
    
    # En passant square
    if epsquare(board) != SQ_NONE
		tile = square_to_int(tostring(epsquare(board)))    
		tensor[18, div(tile - 1, 8) + 1, (tile - 1) % 8 + 1] = 1
    end
    
    return tensor
end


function encode_move(move::String)
	from_s = move[1:2]
	to_s = move[3:4]
	from_tile = square_to_int(from_s)
	to_tile = square_to_int(to_s)
	return UInt16((from_tile - 1) * 64 + to_tile)
end


function square_to_int(tile::String)
	f = tile[1]
    r = tile[2]
    file = 8 - ('h' - f)
    row = 7 - ('8' - r)
    return 8 * row + file
end


function get_game_over_and_value(board::Board)
	if isterminal(board)
		if ischeckmate(board)
			return (true, board.turn == :white ? -1 : 1)
		else
			return (true, 0)
		end
    end
	return (false, 0)
end


function game_over(game::SimpleGame)
    if isterminal(game)
        return true
    end
    return false
end


function int_to_move(move_int::Integer)
    files = "abcdefgh"
    ranks = "12345678"
    
    from_file = files[(move_int - 1) ÷ 64 % 8 + 1]
    from_rank = ranks[(move_int - 1) ÷ 512 + 1]
    to_file = files[(move_int - 1) % 8 + 1]
    to_rank = ranks[(move_int - 1) ÷ 8 % 8 + 1]
    
    return movefromstring(string(from_file, from_rank, to_file, to_rank))
end


function get_legal_moves(board::Board)
	legal_moves = moves(board)
	valid_moves = []
	for move in legal_moves
		new_move = encode_move(tostring(move))
		push!(valid_moves, new_move)
	end
	return valid_moves

end

function get_valid_moves(board::Board)
	valid_moves = zeros(4096)
	legal_moves = moves(board)
	for move in legal_moves
		move_pos = encode_move(tostring(move))
		valid_moves[move_pos] = 1
	end
	return valid_moves
end


function board_to_tensor(board::Board)
	tensor = create_input_tensors(board)
    tensor = Float32.(tensor)
    tensor = reshape(tensor, 1, 18, 8, 8)                                   
	tensor = permutedims(tensor, (3, 4, 2, 1))
	return tensor
end
