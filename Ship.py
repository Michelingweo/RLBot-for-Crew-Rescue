import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import numpy as np
import time
import math
import copy
import pickle


class Ship_node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = 'closed'  # 0:closed, 1: open
        self.if_Crew = 0  # 0: no Crew, 1: have Crew
        self.if_Alien = 0  # 0: no Alien, 1: have Alien
        self.if_Bot = 0  # 0: no Bot, 1: have Bot
        self.was_Crew = 0 # 0: never be a crew; 1: used to be a crew
        self.if_visited = 0


class Agent:

    def __init__(self, Ship_Dimention):

        self.x = None
        self.y = None
        self.Ship_Dimention = Ship_Dimention
        self.move_direction_list = ['up', 'down', 'left', 'right']

    def is_valid(self, x, y, ship_matrix):

        return 0 <= x < self.Ship_Dimention and 0 <= y < self.Ship_Dimention and ship_matrix[x][y].state == 'open'

    def move(self, move_direction, Ship_matrix):

        if move_direction == 'up':
            if self.is_valid(self.x, self.y + 1, Ship_matrix):
                self.y += 1
                return True
            else:
               return False
        elif move_direction == 'down':
            if self.is_valid(self.x, self.y - 1, Ship_matrix):
                self.y -= 1
                return True
            else:
                return False 
        elif move_direction == 'left':
            if self.is_valid(self.x - 1, self.y, Ship_matrix):
                self.x -= 1
                return True
            else:
                return False      
        elif move_direction == 'right':
            if self.is_valid(self.x + 1, self.y, Ship_matrix):
                self.x += 1
                return True
            else:
                return False
        # elif move_direction == 'stay':
        #     return True
        else:
            raise("Wrong Move direction")
    
    def random_move(self, Ship_matrix):

        while True:
            random_direction = random.choice(self.move_direction_list)
            if self.move(random_direction, Ship_matrix):
                return True


class Ship:

    def __init__(self, Ship_Dimention, Num_of_Crew, Num_of_Alien, K, alpha):

        self.Num_of_Crew = Num_of_Crew
        self.Num_of_Alien = Num_of_Alien
        self.Ship_Dimention = Ship_Dimention
        self.K = K  # The Alien must be initializd outside the (2k + 1) × (2k + 1) of the initial bot position
        self.alpha = alpha
        print('-----Initialized the Ship-----')
        print('Ship_Dimention = ', Ship_Dimention)
        print('Num_of_Crew = ', Num_of_Crew)
        print('Num_of_Alien = ', Num_of_Alien)
        print('K = ', K)
        print('alpha=', alpha)
        

        self.Ship_matrix = [[Ship_node(x, y) for x in range(self.Ship_Dimention)]
                            for y in range(self.Ship_Dimention)]
        
        
        self.Create_Ship()
        self.Original_Ship_matrix = copy.deepcopy(self.Ship_matrix)
        self.Put_crew_alien_bot_in_Ship()
        self.Gen_Shortest_Path_Dic()
        self.Initialize_node_transfer_prob_matrix()
        self.last_bot_position = None
        print('-----Successfully Initialized the Ship-----')
        
    def is_valid(self, x, y):

        return 0 <= x < self.Ship_Dimention and 0 <= y < self.Ship_Dimention

    def Create_Ship(self):

        # Choose a random square in the interior to open
        start_x, start_y = random.randint(1, self.Ship_Dimention - 2), random.randint(
            1, self.Ship_Dimention - 2)
        self.Ship_matrix[start_x][start_y].state = 'open'

        while True:

            blocked_node_with_one_open_neighbor = [
                (i, j) for i in range(0, self.Ship_Dimention) for j in range(0, self.Ship_Dimention)
                if self.Ship_matrix[i][j].state == 'closed'
                and self.Count_open_neighbors(i, j) == 1
            ]
            if blocked_node_with_one_open_neighbor != []:
                # randomly select a blocked node with one open neighbor
                x, y = random.choice(blocked_node_with_one_open_neighbor)
                self.Ship_matrix[x][y].state = 'open'
            else:
                break

        dead_ends = [(i, j) for i in range(0, self.Ship_Dimention)
                        for j in range(0, self.Ship_Dimention)
                        if self.Ship_matrix[i][j].state == 'open'
                        and self.Count_open_neighbors(i, j) == 1]

        for _ in range(len(dead_ends) // 2):
            # for (x,y) in dead_ends:
            x, y = random.choice(dead_ends)
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            neighbors = [(i, j) for i, j in neighbors if self.is_valid(i, j)
                            and self.Ship_matrix[i][j].state == 'closed']

            if neighbors:
                nx, ny = random.choice(neighbors)
                self.Ship_matrix[nx][ny].state = 'open'

        open_nodes_num = 0
        open_nodes_position_list = []
        for x in range(self.Ship_Dimention):

            for y in range(self.Ship_Dimention):

                if self.Ship_matrix[x][y].state == 'open':

                    open_nodes_num += 1
                    open_nodes_position_list.append((x, y))


        self.open_nodes_num = open_nodes_num
        self.open_nodes_postion_list = open_nodes_position_list

        self.original_Ship_state_matrix = [[0 for _ in range(self.Ship_Dimention)]
                                  for _ in range(self.Ship_Dimention)]

        for x in range(self.Ship_Dimention):

            for y in range(self.Ship_Dimention):

                if self.Ship_matrix[x][y].state == 'open':
                    
                    self.original_Ship_state_matrix[x][y] = 1



        # self.Show_Ship()

    def Count_open_neighbors(self, x, y):
        count = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in directions:
            if self.is_valid(
                    x + dx, y +
                    dy) and self.Ship_matrix[x + dx][y + dy].state == 'open':
                count += 1

        return count

    def Put_crew_alien_bot_in_Ship(self):

        # randomly select open node to put one bot
        bot = Agent(self.Ship_Dimention)
        while (True):
            bot_x, bot_y = random.randint(0, self.Ship_Dimention - 1), random.randint(0, self.Ship_Dimention - 1)
            if self.Ship_matrix[bot_x][bot_y].state == 'open' and self.Ship_matrix[bot_x][bot_y].if_Alien != 1:
                self.Ship_matrix[bot_x][bot_y].if_Bot = 1
                bot.x = bot_x
                bot.y = bot_y
                break
            else:
                continue

        self.bot = bot

        # create alien instances based on Agent
        alien_list = [Agent(self.Ship_Dimention) for _ in range(self.Num_of_Alien)]
        succeed_alien_num = 0
        # randomly put alien on the ship if the node is open
        while (succeed_alien_num < self.Num_of_Alien):
            x, y = random.randint(0, self.Ship_Dimention - 1), random.randint(0, self.Ship_Dimention - 1)

            if self.Ship_matrix[x][y].state == 'open' and \
                self.Ship_matrix[x][y].if_Alien == 0 and\
                   not self.If_in_DS((x, y)):

                self.Ship_matrix[x][y].if_Alien = 1
                alien_list[succeed_alien_num].x = x
                alien_list[succeed_alien_num].y = y
                succeed_alien_num += 1
            else:
                continue

        self.alien_list = alien_list

        # create crew instances based on Agent
        crew_list = [Agent(self.Ship_Dimention) for _ in range(self.Num_of_Crew)]
        succeed_crew_num = 0
        # randomly put crew on the ship if the node is open
        while (succeed_crew_num < self.Num_of_Crew):
            x, y = random.randint(0, self.Ship_Dimention - 1), random.randint(0, self.Ship_Dimention - 1)

            if self.Ship_matrix[x][y].state == 'open' and \
                self.Ship_matrix[x][y].if_Bot == 0:

                self.Ship_matrix[x][y].if_Crew = 1
                crew_list[succeed_crew_num].x = x
                crew_list[succeed_crew_num].y = y
                succeed_crew_num += 1
            else:
                continue

        self.crew_list = crew_list
                
        return

    def Show_Ship(self):

        custom_cmap = ListedColormap(
            ['black', 'white', 'blue', 'red', 'green'])

        self.Ship_state_matrix = [[0 for _ in range(self.Ship_Dimention)]
                                  for _ in range(self.Ship_Dimention)]

        for x in range(self.Ship_Dimention):

            for y in range(self.Ship_Dimention):

                if self.Ship_matrix[x][y].state == 'open':
                    if self.Ship_matrix[x][y].if_Bot == 1:
                        self.Ship_state_matrix[x][y] = 2
                    
                    elif self.Ship_matrix[x][y].if_Alien == 1:
                        self.Ship_state_matrix[x][y] = 3

                    elif self.Ship_matrix[x][y].if_Crew == 1:
                        self.Ship_state_matrix[x][y] = 4

                    else:
                        self.Ship_state_matrix[x][y] = 1
                else:
                    continue

        self.Ship_state_matrix = np.array(self.Ship_state_matrix)
        self.Ship_state_matrix = np.transpose(self.Ship_state_matrix)

        print(self.Ship_state_matrix)

        plt.figure(figsize=(self.Ship_Dimention, self.Ship_Dimention))
        plt.imshow(self.Ship_state_matrix, cmap=custom_cmap, origin='lower')
 
        plt.show()

    def Get_Ship_and_bot_matrix(self):

        matrix = [[0 for _ in range(self.Ship_Dimention)]
                                for _ in range(self.Ship_Dimention)]

        for x in range(self.Ship_Dimention):

            for y in range(self.Ship_Dimention):

                if self.Ship_matrix[x][y].state == 'open':

                    if self.Ship_matrix[x][y].if_Bot == 1:
                        matrix[x][y] = 2
                    else:
                        matrix[x][y] = 1
                        
                else:
                    continue

        return matrix

    def If_in_DS(self, position):
            DS = self.DS()
            if DS[0][0] <= position[0] <= DS[0][1] and DS[1][0] <= position[1] <= DS[1][1]:
                
                return True
            
            else:
                return False
            
    def DS_open_nodes_list(self):

        DS_open_nodes_list = []

        for open_node_position in self.open_nodes_postion_list:

            if self.If_in_DS(open_node_position):

                DS_open_nodes_list.append(open_node_position)

        return DS_open_nodes_list

    def Alien_Detection(self):

        DS = self.DS()

        for x in range(DS[0][0],DS[0][1]+1):

            for y in range(DS[1][0],DS[1][1]+1):

                if self.Ship_matrix[x][y].if_Alien == 1:

                    return True
                
        return False
    
    def DS(self):

        return [(max(0, self.bot.x - self.K), min(self.Ship_Dimention-1, self.bot.x + self.K)),
                (max(0, self.bot.y - self.K), min(self.Ship_Dimention-1, self.bot.y + self.K))]
    
    def Compute_Shortest_Path(self, bot_pos):

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n = self.Ship_Dimention
        paths = {}
        visited = [[False] * n for _ in range(n)]
        distance = {(x, y): float('inf') for x in range(n) for y in range(n)}
        
        # Queue for BFS
        queue = deque([(bot_pos, [bot_pos])])
        visited[bot_pos[0]][bot_pos[1]] = True
        distance[bot_pos] = 0
        
        while queue:
            (x, y), path = queue.popleft()
            
            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny] and self.Ship_matrix[nx][ny].state == 'open':
                    visited[nx][ny] = True
                    paths[(nx, ny)] = path + [(nx, ny)]
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    distance[(nx, ny)] = distance[(x, y)] + 1
        
        return distance, paths

    def Gen_Shortest_Path_Dic(self):
        
        self.Shortest_Path_Dic = dict()
        self.Shortest_Distance_Dic = dict()
        
        for x in range(self.Ship_Dimention):
            for y in range(self.Ship_Dimention):
                if self.Ship_matrix[x][y].state == 'open':
                    cur_pos = (x,y)
                    distances_, paths_ = self.Compute_Shortest_Path(cur_pos) # return a dictionary
                    self.Shortest_Distance_Dic[cur_pos] = distances_
                    self.Shortest_Path_Dic[cur_pos] = paths_
                else:
                    continue
        return
    
    def Get_SDD_reverse(self):
        
        self.SDD_reverse = dict() # SDD_reverse = {bot_pos:{distance:[crew_pos]}}
        for bot_pos in self.Shortest_Distance_Dic.keys():
            distance_dict_ = set()
            for crew_pos, d in self.Shortest_Distance_Dic[bot_pos]:
                if d not in distance_dict_:
                    distance_dict_[d] = {crew_pos}
                else:
                    distance_dict_[d].add(crew_pos)
            
            self.SDD_reverse[bot_pos] = distance_dict_
        
        return
    
    def Beep_Detection(self):
        
        crew_pos_list = [(crew.x,crew.y) for crew in self.crew_list]
        bot_pos = (self.bot.x, self.bot.y)
        # probability of not receive a beep
        total_P = 0.0
        for crew_pos in crew_pos_list:
            # print("bot_pos:", bot_pos)
            # print('crew_pos:', crew_pos)
            try:
                d_ = self.Shortest_Distance_Dic[bot_pos][crew_pos]
            except:
                print("if bot closed:", self.Ship_matrix[bot_pos[0]][bot_pos[1]].state)
                
            total_P *= 1.0 - math.exp(-self.alpha * (d_ - 1))
        
        return np.random.choice([True,False],p=[1-total_P,total_P])

    def Bot_move(self, new_pos):
        self.Ship_matrix[self.bot.x][self.bot.y].if_Bot = 0

        self.last_bot_position = (self.bot.x, self.bot.y)

        self.bot.x = new_pos[0]
        self.bot.y = new_pos[1]
        self.Ship_matrix[new_pos[0]][new_pos[1]].if_Bot = 1

        self.Ship_matrix[new_pos[0]][new_pos[1]].if_visited = 1

    def Alien_Random_move(self):

        alien_pos = []
        for alien in self.alien_list:  

            # remove original indicator
            self.Ship_matrix[alien.x][alien.y].if_Alien = 0

            #random move
            alien.random_move(self.Ship_matrix)

            #new indicator
            self.Ship_matrix[alien.x][alien.y].if_Alien = 1

        return alien_pos
    
    def Initialize_node_transfer_prob_matrix(self):

        self.node_transfer_prob_matrix = np.zeros((self.Ship_Dimention * self.Ship_Dimention, self.Ship_Dimention * self.Ship_Dimention))
   
        self.neighbor_dict = {}
        self.neighbor_open_direction_dict = {}

        for position in self.open_nodes_postion_list:

            x = position[0]
            y = position[1]

            directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

            cell_i_index = x * self.Ship_Dimention + y

            neighbor_num = 0

            neighbor_nodes_list = []
            neighbor_nodes_direction_list = [0, 0, 0, 0]

            for direction in range(4):
                dx, dy = directions[direction]

                if self.is_valid(x + dx, y +dy) and self.Ship_matrix[x + dx][y + dy].state == 'open':

                    neighbor_nodes_list.append((x + dx, y +dy))

                    neighbor_nodes_direction_list[direction] = 1
                    neighbor_num += 1
                
            
            self.neighbor_dict[(x, y)] = neighbor_nodes_list
            self.neighbor_open_direction_dict[(x, y)] = neighbor_nodes_direction_list

            for dx, dy in directions:

                if self.is_valid(x + dx, y +dy) and self.Ship_matrix[x + dx][y + dy].state == 'open' and neighbor_num > 0:

                    cell_j_index = dx * self.Ship_Dimention + dy
                    self.node_transfer_prob_matrix[cell_i_index][cell_j_index] = 1/neighbor_num 

    def Get_open_neighbor_position_list(self, position):

        return self.neighbor_dict[position]

    def Prob_Move_to_DS(self, position):

        Neighbor_nodes_list = self.Get_open_neighbor_position_list(position)

        num = 0

        for (x, y) in Neighbor_nodes_list:

            if self.If_in_DS((x, y)):
                num += 1

        if len(Neighbor_nodes_list) == 0:
            return 0
        
        return num/len(Neighbor_nodes_list)
    
    def Prob_Move_to_aim_position(self, original_position, aim_position):

        Neighbor_nodes_list = self.Get_open_neighbor_position_list(original_position)

        num = 0

        for (x, y) in Neighbor_nodes_list:

            if (x,y) == aim_position:
                num += 1

        if len(Neighbor_nodes_list) == 0:
            return 0
        
        return num/len(Neighbor_nodes_list)
     
    def Reset_Ship(self, K, alpha, deep_reset = False):

        if deep_reset:

            self.Ship_matrix = [[Ship_node(x, y) for x in range(self.Ship_Dimention)]
                                for y in range(self.Ship_Dimention)]
            
            
            self.Create_Ship()
            self.Original_Ship_matrix = copy.deepcopy(self.Ship_matrix)
            self.Put_crew_alien_bot_in_Ship()
            self.Gen_Shortest_Path_Dic()
            self.Initialize_node_transfer_prob_matrix()
            self.K = K
            self.alpha = alpha
            self.last_bot_position = None

        else:

            # reset the position of alien, crew and bot
            self.Ship_matrix = copy.deepcopy(self.Original_Ship_matrix)

            self.Put_crew_alien_bot_in_Ship()

            self.K = K

            self.alpha = alpha

            self.last_bot_position = None

    def Rescue_Update_Crew(self):

        bot_x = self.bot.x
        bot_y = self.bot.y

        new_crew_list = []
        if_rescue_one = False
        
        for crew in self.crew_list:

            if crew.x != bot_x or crew.y != bot_y:
                # this crew is not be rescued yet.
                new_crew_list.append(crew)

            else:
                if_rescue_one = True
                # this crew is rescued, update the ship coordinate property
                self.Ship_matrix[crew.x][crew.y].if_Crew = 0
                self.Ship_matrix[crew.x][crew.y].was_Crew = 1      
        
        self.crew_list = new_crew_list
        
        return if_rescue_one

    def Over(self):
        # if the game is over?
        bot_x = self.bot.x
        bot_y = self.bot.y
        
        for alien in self.alien_list:

            if alien.x == bot_x and alien.y == bot_y:
                # if bot be caught by alien
                return True, 'Fail'
            
            if len(self.crew_list) == 0:
                # if rescued all the crew
                return True, 'Success'
        
        return False, None

    def Search_block(self, position, block_size):

        search_block = []

        for x in range(max(0, position[0] - block_size), min(self.Ship_Dimention-1, position[0] + block_size) + 1):
        
            for y in range(max(0, position[1] - block_size), min(self.Ship_Dimention-1, position[1]+ block_size) + 1):

                # print("candidate position:", (x, y))
                if self.Ship_matrix[x][y].if_visited == 0 and self.Ship_matrix[x][y].state == "open":

                    search_block.append((x, y))

        return search_block

    def action_one_hot(self, next_step):

        bot_x_change = next_step[0] - self.bot.x
        bot_y_change = next_step[1] - self.bot.y

        if bot_x_change == 0 and bot_y_change == 1: 
            return [1,0,0,0]
        elif bot_x_change == 0 and bot_y_change == -1: 
            return [0,1,0,0]
        elif bot_x_change == 1 and bot_y_change == 0: 
            return [0,0,0,1]
        elif bot_x_change == -1 and bot_y_change == 0: 
            return [0,0,1,0]
        elif bot_x_change == 0 and bot_y_change == 0:
            return [0,0,0,0]
        else:
            raise Exception("Wrong next_step: {}, Bot position: {}".format(next_step, (self.bot.x, self.bot.y)))
        

if __name__ == '__main__':

    Ship_Dimention = 50
    Num_of_Crew = 1
    Num_of_Alien = 1

    Our_Ship = Ship(Ship_Dimention, Num_of_Crew, Num_of_Alien, 3, 1)

    Our_Ship.Show_Ship()

    # 将实例保存到文件
    with open('Ship.pkl', 'wb') as file:
        pickle.dump(Our_Ship, file)
    
    with open('Ship.pkl', 'rb') as file:
        Ship_load = pickle.load(file)

    Ship_load.Show_Ship()