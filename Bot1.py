from collections import deque
import copy
from Ship import *
import numpy as np
from matplotlib import pyplot as plt
import math
from matplotlib.colors import ListedColormap
from collections import defaultdict
import json
import tqdm
import pickle


np.set_printoptions(threshold=np.inf)


class Bot1:
    
    def __init__(self, Ship: Ship):

        self.bot_index = 1

        self.Ship = Ship

        self.Ship_Alien_Prob_matrix = np.zeros((self.Ship.Ship_Dimention, self.Ship.Ship_Dimention))
        self.Ship_Crew_Prob_matrix = np.zeros((self.Ship.Ship_Dimention, self.Ship.Ship_Dimention))

        self.open_nodes_postion_list = self.Ship.open_nodes_postion_list

        self.Initialize_Ship_Alien_and_Crew_Prob_matrix()

        self.image_index = 0


    def Initialize_Ship_Alien_and_Crew_Prob_matrix(self):

        num = 0

        for (x,y) in self.Ship.open_nodes_postion_list:

            if not self.Ship.If_in_DS((x,y)):

                num +=1
            
            if x == self.Ship.bot.x and y == self.Ship.bot.y:
                self.Ship_Crew_Prob_matrix[x][y] = 0
            else:
                self.Ship_Crew_Prob_matrix[x][y] = 1/(self.Ship.open_nodes_num-1)

        for (x,y) in self.Ship.open_nodes_postion_list:

            if not self.Ship.If_in_DS((x,y)):

                self.Ship_Alien_Prob_matrix[x][y] = 1/num

        # self.Show_Ailen_Crew_Prob_Heat_Map()


    def Show_Ailen_Crew_Prob_Heat_Map(self):

        fig, axs = plt.subplots(1, 3, figsize=(self.Ship.Ship_Dimention, self.Ship.Ship_Dimention))
    
        # show ship 
        custom_cmap = ListedColormap(
            ['black', 'white', 'blue', 'red', 'green'])

        Ship_state_matrix = [[0 for _ in range(self.Ship.Ship_Dimention)]
                                  for _ in range(self.Ship.Ship_Dimention)]

        for x in range(self.Ship.Ship_Dimention):

            for y in range(self.Ship.Ship_Dimention):

                if self.Ship.Ship_matrix[x][y].state == 'open':
                    if self.Ship.Ship_matrix[x][y].if_Bot == 1:
                        Ship_state_matrix[x][y] = 2
                    
                    elif self.Ship.Ship_matrix[x][y].if_Alien == 1:
                        Ship_state_matrix[x][y] = 3

                    elif self.Ship.Ship_matrix[x][y].if_Crew == 1:
                        Ship_state_matrix[x][y] = 4

                    else:
                        Ship_state_matrix[x][y] = 1
                else:
                    continue

        Ship_state_matrix = np.array(Ship_state_matrix)
        Ship_state_matrix = np.transpose(Ship_state_matrix)

        if not np.isin(4, Ship_state_matrix):
            custom_cmap = ListedColormap(['black', 'white', 'blue', 'red'])


        axs[0].imshow(Ship_state_matrix, cmap=custom_cmap, origin='lower')

         # prob heat map

        Ship_Alien_Prob_matrix_transposed = np.transpose(self.Ship_Alien_Prob_matrix)

        axs[1].imshow(Ship_Alien_Prob_matrix_transposed, cmap='cividis', interpolation='nearest', origin='lower', vmin=0, vmax=1)
        axs[1].set_title('Heatmap Alien_Prob')

        for i in range(Ship_Alien_Prob_matrix_transposed.shape[0]):
            for j in range(Ship_Alien_Prob_matrix_transposed.shape[1]):
                axs[1].text(j, i, f'{Ship_Alien_Prob_matrix_transposed[i, j]:.5f}', ha='center', va='center', color='w', fontsize=5)


        Ship_Crew_Prob_matrix_transposed = np.transpose(self.Ship_Crew_Prob_matrix)

        axs[2].imshow(Ship_Crew_Prob_matrix_transposed, cmap='cividis', interpolation='nearest', origin='lower', vmin=0, vmax=1)
        axs[2].set_title('Heatmap Crew_Prob')

        for i in range(Ship_Crew_Prob_matrix_transposed.shape[0]):
            for j in range(Ship_Crew_Prob_matrix_transposed.shape[1]):
                axs[2].text(j, i, f'{Ship_Crew_Prob_matrix_transposed[i, j]:.5f}', ha='center', va='center', color='w', fontsize=5)

        
        plt.savefig(f'./result/heatmap{self.image_index}.png')
        self.image_index +=1

        plt.close()

        # 显示图形
        # plt.show()


    def Update_Alien_Prob_After_Bot_Move(self):
        # bot move from cell j to cell i, what will the other cell prob change
        Bot_position = (self.Ship.bot.x, self.Ship.bot.y)
        for (x,y) in self.Ship.open_nodes_postion_list:

            if x == Bot_position[0] and y == Bot_position[1]:

                self.Ship_Alien_Prob_matrix[x][y] = 0

            else:

                P1 = 1
                P2 = self.Ship_Alien_Prob_matrix[x][y]
                P3 =  1 - self.Ship_Alien_Prob_matrix[Bot_position[0]][Bot_position[1]]

                self.Ship_Alien_Prob_matrix[x][y] = P1 * P2/P3 if P3 != 0 else 0

    
    def Update_Alien_Prob_After_Bot_Move_AD(self, AD_result):

        if AD_result:

            DS = self.Ship.DS()
            P3 = 0

            for x in range(DS[0][0],DS[0][1] + 1):

                for y in range(DS[1][0],DS[1][1] + 1):

                    if  self.Ship.Ship_matrix[x][y].state == 'open':

                        P3 += self.Ship_Alien_Prob_matrix[x][y]

            # print(P3)

            for (x,y) in self.Ship.open_nodes_postion_list:

                if self.Ship.If_in_DS((x,y)):

                    # print("in", (x,y))
                    P1 = 1
                else:

                    # print("out", (x,y))
                    P1 = 0



                P2 = self.Ship_Alien_Prob_matrix[x][y]

                self.Ship_Alien_Prob_matrix[x][y] = P1 * P2/P3 if P3 != 0 else 0

        else:

            P3 = 0

            for (x,y) in self.Ship.open_nodes_postion_list:

                if not self.Ship.If_in_DS((x,y)):
                    
                    P3 += self.Ship_Alien_Prob_matrix[x][y]

            for (x,y) in self.Ship.open_nodes_postion_list:

                if self.Ship.If_in_DS((x,y)):
                    P1 = 0
                else:
                    P1 = 1

                P2 = self.Ship_Alien_Prob_matrix[x][y]

                self.Ship_Alien_Prob_matrix[x][y] = P1 * P2/P3 if P3 != 0 else 0  


    def Update_Alien_Prob_After_Alien_Move(self):

        for (x,y) in self.Ship.open_nodes_postion_list:

            Neighbor_nodes_list = self.Ship.Get_open_neighbor_position_list((x,y))

            self.Ship_Alien_Prob_matrix[x][y] = 0
            for (nei_x,nei_y) in Neighbor_nodes_list:
                self.Ship_Alien_Prob_matrix[x][y] += self.Ship_Alien_Prob_matrix[nei_x][nei_y] * self.Ship.Prob_Move_to_aim_position((nei_x,nei_y), (x,y))


    def Update_Alien_Prob_After_Alien_Move_AD(self, AD_result):

        self.Update_Alien_Prob_After_Bot_Move_AD(AD_result)


    def Update_Crew_Prob_After_Bot_Move(self):
        
        Bot_position = (self.Ship.bot.x, self.Ship.bot.y)
        
        P3 = 1 - self.Ship_Crew_Prob_matrix[Bot_position[0]][Bot_position[1]]
        for (x,y) in self.Ship.open_nodes_postion_list:

            P1 = 1

            if (x,y) == Bot_position:
                P1 = 0

            self.Ship_Crew_Prob_matrix[x][y] = P1 * self.Ship_Crew_Prob_matrix[x][y] / P3 if P3 != 0 else 0
        return


    def Update_Crew_Prob_After_BD(self):
        
        for (x,y) in self.Ship.open_nodes_postion_list:
            if (x,y) == (self.Ship.bot.x,self.Ship.bot.y):
                self.Ship_Crew_Prob_matrix[x][y] = 0.0
            else:
                d_ = self.Ship.Shortest_Distance_Dic[(self.Ship.bot.x,self.Ship.bot.y)][(x,y)]
                P1 = math.exp(-self.Ship.alpha * (d_ - 1))
                P2 = self.Ship_Crew_Prob_matrix[x][y]
                if self.Ship_Crew_Prob_matrix[x][y] == 0.0:
                    continue
                else:
                    self.Ship_Crew_Prob_matrix[x][y] = max(P1 * P2, 0.00001)
            
        self.Ship_Crew_Prob_matrix /= np.sum(self.Ship_Crew_Prob_matrix)
        return


    def Next_Bot_Move(self):
        # get bot position
        cur_bot_pos = (self.Ship.bot.x,self.Ship.bot.y)
        
        # get the highest crew probability position
        max_index_flat = np.argmax(self.Ship_Crew_Prob_matrix)
        highest_prob_crew_pos = np.unravel_index(max_index_flat, self.Ship_Crew_Prob_matrix.shape)
        # get the shortest track
        track2crew = self.Ship.Shortest_Path_Dic[cur_bot_pos][highest_prob_crew_pos]
        # print('track:', track2crew)
        if len(track2crew) > 1:
            next_step = track2crew[1]
        else:
            return
        
        # find the alien prob around the bot
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        alien_prob_around = dict()
        for dx, dy in directions:
            nx, ny = cur_bot_pos[0] + dx, cur_bot_pos[1] + dy
            if self.Ship.is_valid(nx, ny) and self.Ship.Ship_matrix[nx][ny].state=='open':
                alien_prob_around[(nx, ny)] = self.Ship_Alien_Prob_matrix[nx][ny]
            else:
                continue
        
        # if alien_prob_around[next_step] > 0:
            # next_step = cur_bot_pos
        
        if alien_prob_around[next_step] == max(alien_prob_around.values()) and max(alien_prob_around.values()) != 0.0:
            for key, value in alien_prob_around.items():
                if value == min(alien_prob_around.values()):
                    next_step = key

        action_one_hot = self.Ship.action_one_hot(next_step=next_step)
        
        self.Ship.Bot_move(new_pos=next_step)
        # self.Show_Ailen_Crew_Prob_Heat_Map()

        return action_one_hot


    def Reset(self, K, alpha):

        print()
        print("Reset the Ship ...", end='  ')
        # print("K:", K)
        # print("alpha:", alpha)

        self.Ship.Reset_Ship(K, alpha, deep_reset=False)

        self.Ship_Alien_Prob_matrix = np.zeros((self.Ship.Ship_Dimention, self.Ship.Ship_Dimention))
        self.Ship_Crew_Prob_matrix = np.zeros((self.Ship.Ship_Dimention, self.Ship.Ship_Dimention))

        self.open_nodes_postion_list = self.Ship.open_nodes_postion_list

        self.Initialize_Ship_Alien_and_Crew_Prob_matrix()

        # self.Show_Ailen_Crew_Prob_Heat_Map()

        print("Reset success!")


    def Run(self, K_list, alpha_list, num_trails=100):
        
        for K in K_list:

            probability_success_list = [] 
            average_step_num_to_save_all_crews_list = []
            average_crew_num_saved_list = []

            for alpha in alpha_list:
                
                self.K = K
                self.alpha = alpha
                
                success_list = []
                step_num_to_save_all_crews_list = []
                crew_num_saved_list = []

                for i in range(num_trails):

                    self.Reset(K, alpha)

                    step_num = 0
                    crew_save_num = 0

                    while True:
                        # print(self.Ship.bot.x, self.Ship.bot.y)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        
                        self.Next_Bot_Move()

                        # print(self.Ship.bot.x, self.Ship.bot.y)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        step_num += 1

                        Over, Final =  self.Ship.Over()

                        if Over:
                            print("Over! Task:", Final)
                          
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            
                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        if self.Ship.Rescue_Update_Crew():
                            crew_save_num += 1
                            print(f'A crew is rescued! Number of moves: {step_num}')

                        self.Update_Alien_Prob_After_Bot_Move()
                        # print('Update_Alien_Prob_After_Bot_Move')
                        
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Update_Crew_Prob_After_Bot_Move()
                        # print('Update_Crew_Prob_After_Bot_Move')
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)

                        self.Update_Alien_Prob_After_Bot_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        
                        BD_result = self.Ship.Beep_Detection()

                        if BD_result:
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            self.Update_Crew_Prob_After_BD()
                            # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Ship.Alien_Random_move()

                        Over, Final =  self.Ship.Over()

                        if Over:
                            
                            print("Over! Task:", Final)
                          

                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        self.Update_Alien_Prob_After_Alien_Move()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)
                        
                        self.Update_Alien_Prob_After_Alien_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                print("alpha:", alpha)
                print("success_list: ", success_list)
                print("step_num_to_save_all_crews_list: ", step_num_to_save_all_crews_list)
                print("crew_num_saved_list: ", crew_num_saved_list)
                print("================================================================")

                probability_success_list.append(sum(success_list) / len(success_list))
                average_step_num_to_save_all_crews_list.append(np.mean(step_num_to_save_all_crews_list))
                average_crew_num_saved_list.append(np.mean(crew_num_saved_list))


            print('K:', K, 'alpha:', alpha)
            print('probability_success_list:', probability_success_list)
            print('average_step_num_to_save_all_crews_list:', average_step_num_to_save_all_crews_list)
            print('average_crew_num_saved_list:', average_crew_num_saved_list)

            plt.figure(figsize=(20, 6))
            plt.subplot(1, 3, 1)

            plt.plot(alpha_list, average_step_num_to_save_all_crews_list, marker='o', linestyle='-', color='green')
            plt.title(f'Average number of moves needed to \nrescue all crew members with K = {K}')
            plt.xlabel('Alpha')
            plt.ylabel('Number of moves')
            plt.grid(True)

            plt.subplot(1, 3, 2)

            plt.plot(alpha_list, probability_success_list, marker='o', linestyle='-', color='blue')
            plt.title(f'Probability of successfully avoiding the\n alien and rescuing all crew with K = {K}')
            plt.xlabel('Alpha')
            plt.ylabel('Probability')
            plt.grid(True)

            plt.subplot(1, 3, 3)

            plt.plot(alpha_list, average_crew_num_saved_list, marker='o', linestyle='-', color='red')
            plt.title(f'Average number of crew members saved\n with K = {K}')
            plt.xlabel('Alpha')
            plt.ylabel('Number of crew members saved')
            plt.grid(True)

            # # save
            plt.savefig('bot' + str(self.bot_index) + '_result_K_{}.png'.format(K))

            # plt.show()


class Bot1_Proj3(Bot1):
    
    def __init__(self, Ship: Ship):

        super().__init__(Ship)


    def Get_neighbor_Alien_pro(self):

        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        x = self.Ship.bot.x
        y = self.Ship.bot.y

        neighbor_Alien_pro_list = [0, 0, 0, 0]

        for direction in range(4):
            dx, dy = directions[direction]

            if self.Ship.is_valid(x + dx, y +dy) and self.Ship.Ship_matrix[x + dx][y + dy].state == 'open':

                neighbor_Alien_pro_list[direction] = self.Ship_Alien_Prob_matrix[x + dx][y + dy]

        return neighbor_Alien_pro_list


    def Get_next_step_by_onehot_action(self, onehot_action):

        open_neighbor_direction = self.Ship.neighbor_open_direction_dict[(self.Ship.bot.x, self.Ship.bot.y)]

        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        for i in range(len(open_neighbor_direction)):

            if open_neighbor_direction[i] == 0:

                onehot_action[i] = 0

        max_prob_action_direction = directions[[onehot_action.index(max(open_neighbor_direction))][0]]

        next_step = [self.Ship.bot.x + max_prob_action_direction[0], self.Ship.bot.y + max_prob_action_direction[1]]

        return next_step
    

    def Run_and_save_matrix(self, K_list, alpha_list, num_trails=100):

        dataset_save_path = "./dataset_matrix/dataset_episode_"
        
        for K in K_list:

            for alpha in alpha_list:
                
                self.K = K
                self.alpha = alpha
                
                success_list = []
                step_num_to_save_all_crews_list = []
                crew_num_saved_list = []

                for i in tqdm.tqdm(range(num_trails), desc="Sampling Process", unit='episode'):

                    episode_data = []

                    episode = str(i)

                    episode_dataset_save_path = dataset_save_path + episode + ".txt"

                    self.Reset(K, alpha)

                    step_num = 0
                    crew_save_num = 0

                    while True:

                        step_data = []
                        # print(self.Ship.bot.x, self.Ship.bot.y)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        step_data.append(self.Ship_Alien_Prob_matrix.tolist())
                        step_data.append(self.Ship_Crew_Prob_matrix.tolist())
                        step_data.append(self.Ship.Get_Ship_and_bot_matrix())
                        
                        action_onehot = self.Next_Bot_Move()

                        step_data = [step_data, action_onehot]
                        episode_data.append(step_data)

                        # print(self.Ship.bot.x, self.Ship.bot.y)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        step_num += 1

                        Over, Final =  self.Ship.Over()

                        if Over:
                            print("Over! Task:", Final, "Step:", step_num)
                          
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            
                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        if self.Ship.Rescue_Update_Crew():
                            crew_save_num += 1
                            print(f'A crew is rescued! Number of moves: {step_num}')

                        self.Update_Alien_Prob_After_Bot_Move()
                        # print('Update_Alien_Prob_After_Bot_Move')
                        
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Update_Crew_Prob_After_Bot_Move()
                        # print('Update_Crew_Prob_After_Bot_Move')
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)

                        self.Update_Alien_Prob_After_Bot_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        
                        BD_result = self.Ship.Beep_Detection()

                        if BD_result:
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            self.Update_Crew_Prob_After_BD()
                            # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Ship.Alien_Random_move()

                        Over, Final =  self.Ship.Over()

                        if Over:
                            
                            print("Over! Task:", Final, "Step:", step_num)
                          

                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        self.Update_Alien_Prob_After_Alien_Move()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)
                        
                        self.Update_Alien_Prob_After_Alien_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                    # # save the dataset to json
                    # with open(dataset_save_path, 'a') as f:
                    #     json.dumps(dataset_dict[episode], f)

                    # print(dataset_dict[episode])

                    episode_data.append(success_list[-1])

                    with open(episode_dataset_save_path, 'w') as f:
                        for row in episode_data:
                            f.write(str(row) + '\n')  # 写入每行并换行

                print("alpha:", alpha)
                print("K: ", K)
                print("success_list: ", success_list)
                print("step_num_to_save_all_crews_list: ", step_num_to_save_all_crews_list)
                print("crew_num_saved_list: ", crew_num_saved_list)
                print("================================================================")


    def Run_and_save_feature(self, K_list, alpha_list, num_trails=100):

        dataset_save_path = "./dataset_feature/dataset_episode_"
        
        for K in K_list:

            for alpha in alpha_list:
                
                self.K = K
                self.alpha = alpha
                
                success_list = []
                step_num_to_save_all_crews_list = []
                crew_num_saved_list = []

                for i in tqdm.tqdm(range(num_trails), desc="Sampling Process", unit='episode'):

                    episode = str(i)

                    episode_dataset_save_path = dataset_save_path + episode + ".txt"

                    self.Reset(K, alpha)

                    step_num = 0
                    crew_save_num = 0

                    episode_data = []

                    while True:

                        step_data = []

                        # open_neighbor_nodes_direction_list, like [1, 0, 0, 1]
                        step_data.append(self.Ship.neighbor_open_direction_dict[(self.Ship.bot.x, self.Ship.bot.y)])

                        # bot position
                        step_data.append([self.Ship.bot.x, self.Ship.bot.y])

                        # argmax self.Ship_Crew_Prob_matrix position
                        # 获取最大元素的索引
                        flat_index = np.argmax(self.Ship_Crew_Prob_matrix)  # 找到平铺后的最大值索引
                        max_index = np.unravel_index(flat_index, self.Ship_Crew_Prob_matrix.shape) # 将平铺索引转换为原始数组的坐标
                        step_data.append(list(max_index))

                        # DS area alien prob
                        step_data.append(self.Get_neighbor_Alien_pro())

                        # 使用列表推导式展开为一维列表
                        step_data = [element for sublist in step_data for element in sublist]
                        
                        action_onehot = self.Next_Bot_Move()

                        step_data = [step_data, action_onehot]

                        # print("step_data:", step_data)

                        episode_data.append(step_data)

                        # print(self.Ship.bot.x, self.Ship.bot.y)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        step_num += 1

                        Over, Final =  self.Ship.Over()

                        if Over:
                            print("Over! Task:", Final, "Step:", step_num)
                          
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            
                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        if self.Ship.Rescue_Update_Crew():
                            crew_save_num += 1
                            print(f'A crew is rescued! Number of moves: {step_num}')

                        self.Update_Alien_Prob_After_Bot_Move()
                        # print('Update_Alien_Prob_After_Bot_Move')
                        
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Update_Crew_Prob_After_Bot_Move()
                        # print('Update_Crew_Prob_After_Bot_Move')
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)

                        self.Update_Alien_Prob_After_Bot_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        
                        BD_result = self.Ship.Beep_Detection()

                        if BD_result:
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            self.Update_Crew_Prob_After_BD()
                            # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Ship.Alien_Random_move()

                        Over, Final =  self.Ship.Over()

                        if Over:
                            
                            print("Over! Task:", Final, "Step:", step_num)
                          

                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        self.Update_Alien_Prob_After_Alien_Move()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)
                        
                        self.Update_Alien_Prob_After_Alien_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                    # # save the dataset to json
                    # with open(dataset_save_path, 'a') as f:
                    #     json.dumps(dataset_dict[episode], f)

                    # print(dataset_dict[episode])

                    episode_data.append(success_list[-1])

                    with open(episode_dataset_save_path, 'w') as f:
                        for row in episode_data:
                            f.write(str(row) + '\n')  # 写入每行并换行


                print("alpha:", alpha)
                print("K: ", K)
                print("success_list: ", success_list)
                print("step_num_to_save_all_crews_list: ", step_num_to_save_all_crews_list)
                print("crew_num_saved_list: ", crew_num_saved_list)
                print("================================================================")


    def Sample_and_save_feature(self, actor, critic, K_list, alpha_list, num_trails=100):

        replay_buffer_save_path = "./replay_buffer/dataset_episode_"
        
        for K in K_list:

            for alpha in alpha_list:
                
                self.K = K
                self.alpha = alpha
                
                success_list = []
                step_num_to_save_all_crews_list = []
                crew_num_saved_list = []

                for i in tqdm.tqdm(range(num_trails), desc="Sampling Process", unit='episode'):

                    episode = str(i)
                    episode_dataset_save_path = replay_buffer_save_path + episode + ".txt"

                    self.Reset(K, alpha)

                    step_num = 0
                    crew_save_num = 0

                    episode_data = []

                    while True:

                        step_data = []

                        # open_neighbor_nodes_direction_list, like [1, 0, 0, 1]
                        step_data.append(self.Ship.neighbor_open_direction_dict[(self.Ship.bot.x, self.Ship.bot.y)])

                        # bot position
                        step_data.append([self.Ship.bot.x, self.Ship.bot.y])

                        # argmax self.Ship_Crew_Prob_matrix position
                        # 获取最大元素的索引
                        flat_index = np.argmax(self.Ship_Crew_Prob_matrix)  # 找到平铺后的最大值索引
                        max_index = np.unravel_index(flat_index, self.Ship_Crew_Prob_matrix.shape) # 将平铺索引转换为原始数组的坐标
                        step_data.append(list(max_index))

                        # DS area alien prob
                        step_data.append(self.Get_neighbor_Alien_pro())

                        # 使用列表推导式展开为一维列表
                        step_data = [element for sublist in step_data for element in sublist]
                        
                        onehot_action_pred = actor.predict(step_data)
                        next_step = self.Get_next_step_by_onehot_action(onehot_action_pred)

                        self.Ship.Bot_move(new_pos=next_step)

                        critic_best_action = critic.predict(step_data)

                        step_data = [step_data, onehot_action_pred, critic_best_action]

                        # print("step_data:", step_data)

                        episode_data.append(step_data)

                        # print(self.Ship.bot.x, self.Ship.bot.y)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        step_num += 1

                        Over, Final =  self.Ship.Over()

                        if Over:
                            print("Over! Task:", Final, "Step:", step_num)
                          
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            
                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        if self.Ship.Rescue_Update_Crew():
                            crew_save_num += 1
                            print(f'A crew is rescued! Number of moves: {step_num}')

                        self.Update_Alien_Prob_After_Bot_Move()
                        # print('Update_Alien_Prob_After_Bot_Move')
                        
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Update_Crew_Prob_After_Bot_Move()
                        # print('Update_Crew_Prob_After_Bot_Move')
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)

                        self.Update_Alien_Prob_After_Bot_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()
                        
                        BD_result = self.Ship.Beep_Detection()

                        if BD_result:
                            
                            # self.Show_Ailen_Crew_Prob_Heat_Map()
                            self.Update_Crew_Prob_After_BD()
                            # self.Show_Ailen_Crew_Prob_Heat_Map()

                        self.Ship.Alien_Random_move()

                        Over, Final =  self.Ship.Over()

                        if Over:
                            
                            print("Over! Task:", Final, "Step:", step_num)
                          

                            crew_num_saved_list.append(crew_save_num)
                            if Final == 'Fail':
                                success_list.append(0)
                            elif Final == 'Success':
                                success_list.append(1)
                                step_num_to_save_all_crews_list.append(step_num)
                            else:
                                raise("Wrong Final")

                            break

                        self.Update_Alien_Prob_After_Alien_Move()

                        AD_result = self.Ship.Alien_Detection()
                        # print(AD_result)
                        
                        self.Update_Alien_Prob_After_Alien_Move_AD(AD_result)
                        # self.Show_Ailen_Crew_Prob_Heat_Map()

                    episode_data.append(success_list[-1])

                    with open(episode_dataset_save_path, 'w') as f:
                        for row in episode_data:
                            f.write(str(row) + '\n')  # 写入每行并换行


                print("alpha:", alpha)
                print("K: ", K)
                print("success_list: ", success_list)
                print("step_num_to_save_all_crews_list: ", step_num_to_save_all_crews_list)
                print("crew_num_saved_list: ", crew_num_saved_list)
                print("================================================================")



if __name__ == '__main__':
    
    Ship_Dimention = 50
    Bot_idx = 1
    Num_trails = 10 # number of games
    print('Bot:', Bot_idx)
    print('num trails:', Num_trails)

    K_list = [5]
    alpha_list = [0.1]
    
    if Bot_idx == 1:
        
        with open('Ship.pkl', 'rb') as file:
            Ship_load = pickle.load(file)

        Bot = Bot1_Proj3(Ship_load)
        Bot.Run_and_save_matrix(K_list=K_list, alpha_list=alpha_list, num_trails=Num_trails)
        


