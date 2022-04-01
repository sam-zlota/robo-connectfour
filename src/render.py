import pygame
import numpy as np


def render(self, mode: str = 'console', close: bool = False):
    if mode == 'console':
        replacements = {
            self.__player_color: 'A',
            0: ' ',
            -1 * self.__player_color: 'B'
        }

        def render_line(line):
            return "|" + "|".join(
                ["{:>2} ".format(replacements[x]) for x in line]) + "|"

        hline = '|---+---+---+---+---+---+---|'
        print(hline)
        for line in np.apply_along_axis(render_line,
                                        axis=1,
                                        arr=self.__board):
            print(line)
        print(hline)

    elif mode == 'human':
        if self.__screen is None:
            pygame.init()
            self.__screen = pygame.display.set_mode(
                (round(self.__window_width), round(self.__window_height)))

        if close:
            pygame.quit()

        self.__rendered_board = self._update_board_render()
        frame = self.__rendered_board
        surface = pygame.surfarray.make_surface(frame)
        surface = pygame.transform.rotate(surface, 90)
        self.__screen.blit(surface, (0, 0))

        pygame.display.update()
    else:
        raise error.UnsupportedMode()
    return print(render_board(self.board))


def render_board(board,
                 image_width=512,
                 image_height=512,
                 board_percent_x=0.8,
                 board_percent_y=0.8,
                 items_padding_x=0.05,
                 items_padding_y=0.05,
                 slot_padding_x=0.1,
                 slot_padding_y=0.1,
                 background_color=Color.WHITE,
                 board_color=Color.BLUE,
                 empty_slot_color=Color.WHITE,
                 player1_slot_color=Color.RED,
                 player2_slot_color=Color.YELLOW):
    image = Image.new('RGB', (image_height, image_width), background_color)
    draw = ImageDraw.Draw(image)

    board_width = int(image_width * board_percent_x)
    board_height = int(image_height * board_percent_y)

    padding_x = image_width - board_width
    padding_y = image_height - board_height

    padding_top = padding_y // 2
    padding_bottom = padding_y - padding_top

    padding_left = padding_x // 2
    padding_right = padding_x - padding_left

    draw.rectangle([
        (padding_left, padding_top),
        (image_width - padding_right, image_height - padding_bottom)
    ], fill=board_color)

    padding_left += int(items_padding_x * image_width)
    padding_right += int(items_padding_x * image_width)

    padding_top += int(items_padding_y * image_height)
    padding_bottom += int(items_padding_y * image_height)

    cage_width = int((image_width - padding_left - padding_right) / board.shape[1])
    cage_height = int((image_width - padding_top - padding_bottom) / board.shape[0])

    radius_x = int((cage_width - 2 * int(cage_width * slot_padding_x)) // 2)
    radius_y = int((cage_height - 2 * int(cage_height * slot_padding_y)) // 2)

    slots = []
    for row in range(board.shape[0]):
        for column in range(board.shape[1]):
            player = board[row, column]

            actual_row = board.shape[0] - row - 1
            origin_x = padding_left + int(column * cage_width + cage_width // 2)
            origin_y = padding_top + int(actual_row * cage_height + cage_height // 2)

            slots.append((origin_x, origin_y, player))

    for origin_x, origin_y, player in slots:
        color = empty_slot_color
        if player == 1:
            color = player1_slot_color
        elif player == -1:
            color = player2_slot_color

        draw.ellipse([
            (origin_x - radius_x, origin_y - radius_y),
            (origin_x + radius_x, origin_y + radius_y)
        ], fill=color)

    return np.array(image)
