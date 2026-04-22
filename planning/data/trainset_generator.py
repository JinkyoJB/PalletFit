import random
import numpy as np
import json
import matplotlib.pyplot as plt

class temp_Item:
    def __init__(self, x, y, z, width, height, depth):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.adjacent_items = []

    def volume(self):
        return self.width * self.height * self.depth

    def get_bounds(self):
        return (self.x, self.x+self.width, self.y, self.y+self.height, self.z, self.z+self.depth)

class TrainsetGenerator:
    def __init__(self, bin_size=[10,10,10], init_slice=[2,2,2], margin_x=0, margin_y=0):
        """
        Initializes the TrainsetGenerator with bin dimensions, initial slicing, and margins.

        Parameters:
        - bin_size (list): [width, height, depth] of the bin in mm.
        - init_slice (list): [number of slices along x, y, z axes].
        - margin_x (float): Margin to subtract from each item's width after generation.
        - margin_y (float): Margin to subtract from each item's height after generation.
        """
        self.bin_size = bin_size
        self.bin_width = bin_size[0]
        self.bin_height = bin_size[1]
        self.bin_depth = bin_size[2]

        self.init_slice = init_slice
        self.trainset = []

        self.min_width = None
        self.min_height = None
        self.min_depth = None

        self.margin_x = margin_x
        self.margin_y = margin_y

    def generate_trainset(self, num_merge=5, num_split=5):
        """
        Generates a trainset by creating initial items, performing merges and splits,
        and then applying margins to the final items.

        Parameters:
        - num_merge (int): Number of random merge operations to perform.
        - num_split (int): Number of random split operations to perform.

        Returns:
        - final_list (list): List of item dictionaries ready for JSON serialization.
        """
        self.make_init_items()
        self.update_adjacent_all()

        # Compute minimum dimensions among initial items
        self.compute_min_dimensions()

        # Perform merging
        for _ in range(num_merge):
            if not self.trainset:
                break
            item = random.choice(self.trainset)
            self.random_merge(item)

        # Perform splitting
        for _ in range(num_split):
            if not self.trainset:
                break
            item = random.choice(self.trainset)
            axis = random.choice(['x','y','z'])
            if axis == 'x':
                self.split_x(item)
            elif axis == 'y':
                self.split_y(item)
            else:
                self.split_z(item)
            self.update_adjacent_all()

        # Sort the trainset for consistency
        self.trainset.sort(key=lambda it: (it.z, it.y, it.x))

        # Apply margins to the final items
        self.apply_margins()

        # Prepare final list for JSON output
        final_list = []
        for i, it in enumerate(self.trainset):
            final_list.append({
                "partno": str(i),
                "name": f"item_{i}",
                "objshape": "cube",
                "width": it.width,
                "height": it.height,
                "depth": it.depth,
                "rotation_quat": [0,0,0,1],
                "priority": 7,
                "updown": False,
                "options": {'color':"#14ba5e"},
                "weight": 0,
                "loadbear": 0,
                "unit": "mm",
                "b_position": [it.x, it.y, it.z]
            })
        return final_list

    def compute_min_dimensions(self):
        # Calculate minimum width, height, depth among all items
        if not self.trainset:
            return
        min_w = min(it.width for it in self.trainset)
        min_h = min(it.height for it in self.trainset)
        min_d = min(it.depth for it in self.trainset)
        self.min_width = min_w
        self.min_height = min_h
        self.min_depth = min_d

    # def random_merge(self, item1):
    #     if not item1.adjacent_items:
    #         return False
    #     # Select a random adjacent item
    #     valid_adj = [it for it in item1.adjacent_items if it in self.trainset]
    #     if not valid_adj:
    #         return False
    #     item2 = random.choice(valid_adj)

    #     # Check if merging these two items forms a perfect cuboid
    #     x_min = min(item1.x, item2.x)
    #     y_min = min(item1.y, item2.y)
    #     z_min = min(item1.z, item2.z)
    #     x_max = max(item1.x + item1.width, item2.x + item2.width)
    #     y_max = max(item1.y + item1.height, item2.y + item2.height)
    #     z_max = max(item1.z + item1.depth, item2.z + item2.depth)

    #     merged_vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    #     if merged_vol != item1.volume() + item2.volume():
    #         # Not a perfect cuboid after merging
    #         return False

    #     self.merge_items(item1, item2)
    #     return True

    def random_merge(self, item1):
        if not item1.adjacent_items:
            return False

        # x축 또는 y축을 따라 인접한 아이템만 필터링
        adjacent_x = []
        adjacent_y = []
        for it in item1.adjacent_items:
            if it not in self.trainset:
                continue
            # x축을 따라 인접한 경우
            if (item1.x + item1.width == it.x or it.x + it.width == item1.x) and item1.y == it.y and item1.z == it.z:
                adjacent_x.append(it)
            # y축을 따라 인접한 경우
            if (item1.y + item1.height == it.y or it.y + it.height == item1.y) and item1.x == it.x and item1.z == it.z:
                adjacent_y.append(it)

        # 병합 가능한 축 목록 생성
        possible_axes = []
        if adjacent_x:
            possible_axes.append('x')
        if adjacent_y:
            possible_axes.append('y')

        if not possible_axes:
            return False  # 병합 가능한 축이 없으면 종료

        # 병합할 축을 랜덤하게 선택 (x 또는 y)
        axis = random.choice(possible_axes)

        # 선택한 축에 따라 병합할 아이템 선택 및 병합
        if axis == 'x':
            item2 = random.choice(adjacent_x)
            # x축을 따라 병합
            x_min = min(item1.x, item2.x)
            y_min = item1.y
            z_min = item1.z
            x_max = max(item1.x + item1.width, item2.x + item2.width)
            y_max = item1.y + item1.height
            z_max = item1.z + item1.depth
        elif axis == 'y':
            item2 = random.choice(adjacent_y)
            # y축을 따라 병합
            x_min = item1.x
            y_min = min(item1.y, item2.y)
            z_min = item1.z
            x_max = item1.x + item1.width
            y_max = max(item1.y + item1.height, item2.y + item2.height)
            z_max = item1.z + item1.depth

        # 병합된 볼륨 계산
        merged_vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        if merged_vol != item1.volume() + item2.volume():
            # 완전한 직육면체로 병합되지 않으면 병합 실패
            return False

        # 두 아이템을 병합하여 새로운 아이템 생성
        merged = temp_Item(
            x_min, y_min, z_min,
            x_max - x_min, y_max - y_min, z_max - z_min
        )

        # 기존 아이템 제거 및 병합된 아이템 추가
        self.trainset.remove(item1)
        self.trainset.remove(item2)
        self.trainset.append(merged)

        # 인접 아이템 정보 업데이트
        self.update_adjacent_all()

        return True

    def merge_items(self, it1, it2):
        # Merge two items into one
        x_min = min(it1.x, it2.x)
        y_min = min(it1.y, it2.y)
        z_min = min(it1.z, it2.z)
        x_max = max(it1.x + it1.width, it2.x + it2.width)
        y_max = max(it1.y + it1.height, it2.y + it2.height)
        z_max = max(it1.z + it1.depth, it2.z + it2.depth)

        merged = temp_Item(
            x_min, y_min, z_min,
            x_max - x_min, y_max - y_min, z_max - z_min
        )
        self.trainset.remove(it1)
        self.trainset.remove(it2)
        self.trainset.append(merged)
        self.update_adjacent_all()

    def split_x(self, item):
        # Split an item along the x-axis
        if (item.width < 2) or (item.width < self.min_width) or (item.height < self.min_height) or (item.depth < self.min_depth):
            return False
        cut = random.randint(1, item.width - 1)
        it1 = temp_Item(item.x, item.y, item.z, cut, item.height, item.depth)
        it2 = temp_Item(item.x + cut, item.y, item.z, item.width - cut, item.height, item.depth)
        self.trainset.remove(item)
        self.trainset.append(it1)
        self.trainset.append(it2)
        self.update_adjacent_all()
        return True

    def split_y(self, item):
        # Split an item along the y-axis
        if (item.height < 2) or (item.width < self.min_width) or (item.height < self.min_height) or (item.depth < self.min_depth):
            return False
        cut = random.randint(1, item.height - 1)
        it1 = temp_Item(item.x, item.y, item.z, item.width, cut, item.depth)
        it2 = temp_Item(item.x, item.y + cut, item.z, item.width, item.height - cut, item.depth)
        self.trainset.remove(item)
        self.trainset.append(it1)
        self.trainset.append(it2)
        self.update_adjacent_all()
        return True

    def split_z(self, item):
        # Split an item along the z-axis
        if (item.depth < 2) or (item.width < self.min_width) or (item.height < self.min_height) or (item.depth < self.min_depth):
            return False
        cut = random.randint(1, item.depth - 1)
        it1 = temp_Item(item.x, item.y, item.z, item.width, item.height, cut)
        it2 = temp_Item(item.x, item.y, item.z + cut, item.width, item.height, item.depth - cut)
        self.trainset.remove(item)
        self.trainset.append(it1)
        self.trainset.append(it2)
        self.update_adjacent_all()
        return True

    def make_init_items(self):
        """
        Initializes the trainset by dividing the bin into initial slices.
        Margins are not considered during item manipulation; they are applied post-generation.
        """
        x_div = self.init_slice[0]
        y_div = self.init_slice[1]
        z_div = self.init_slice[2]

        def divide_length(total, parts):
            base = total // parts
            remainder = total % parts
            lengths = [base] * parts
            for i in range(remainder):
                lengths[i] += 1
            return lengths

        x_sizes = divide_length(self.bin_width, x_div)
        y_sizes = divide_length(self.bin_height, y_div)
        z_sizes = divide_length(self.bin_depth, z_div)

        x_start = 0
        for x_len in x_sizes:
            y_start = 0
            for y_len in y_sizes:
                z_start = 0
                for z_len in z_sizes:
                    self.trainset.append(temp_Item(
                        x_start, y_start, z_start,
                        x_len, y_len, z_len
                    ))
                    z_start += z_len
                y_start += y_len
            x_start += x_len

    def apply_margins(self):
        """
        Applies margins to all items by subtracting margin_x from width and margin_y from height.
        Ensures that dimensions do not become negative or zero.
        """
        for item in self.trainset:
            # Subtract margins
            new_width = item.width - self.margin_x
            new_height = item.height - self.margin_y

            # Ensure dimensions are not negative or zero
            if new_width <= 0:
                new_width = 1  # Minimum width
            if new_height <= 0:
                new_height = 1  # Minimum height

            item.width = new_width
            item.height = new_height

            # Optionally, adjust position to account for margins
            # Uncomment the following lines if you want to shift items to maintain spacing
            # item.x += self.margin_x / 2
            # item.y += self.margin_y / 2

    def get_adjacent_items(self, item):
        adj = []
        x1_start, x1_end, y1_start, y1_end, z1_start, z1_end = item.get_bounds()

        for other in self.trainset:
            if other is item:
                continue
            x2_start, x2_end, y2_start, y2_end, z2_start, z2_end = other.get_bounds()

            x_shared = ((x1_end == x2_start or x2_end == x1_start)
                        and not (y1_end <= y2_start or y2_end <= y1_start or z1_end <= z2_start or z2_end <= z1_start))
            y_shared = ((y1_end == y2_start or y2_end == y1_start)
                        and not (x1_end <= x2_start or x2_end <= x1_start or z1_end <= z2_start or z2_end <= z1_start))
            z_shared = ((z1_end == z2_start or z2_end == z1_start)
                        and not (x1_end <= x2_start or x2_end <= x1_start or y1_end <= y2_start or y2_end <= y1_start))

            if x_shared or y_shared or z_shared:
                adj.append(other)
        return adj

    def update_adjacent_all(self):
        for it in self.trainset:
            it.adjacent_items = []
        for it in self.trainset:
            it.adjacent_items = self.get_adjacent_items(it)

    def save_trainset_to_json(self, filename, final_list):
        with open(filename, 'w') as f:
            json.dump(final_list, f, indent=4)

    def visualize_trainset(self, filename, save=False):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(0, self.bin_width)
        ax.set_ylim(0, self.bin_height)
        ax.set_zlim(0, self.bin_depth)

        for it in self.trainset:
            self.plot_cube(ax, it.x, it.y, it.z, it.width, it.height, it.depth, color='blue', alpha=0.3)

        plt.title('Trainset Visualization')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        if save:
            plt.savefig(filename)
        plt.show()

    def plot_cube(self, ax, x, y, z, dx, dy, dz, color='red', alpha=0.5):
        xx = [x, x+dx, x+dx, x, x, x+dx, x+dx, x]
        yy = [y, y, y+dy, y+dy, y, y, y+dy, y+dy]
        zz = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]

        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]

        for e in edges:
            ax.plot(
                [xx[e[0]], xx[e[1]]],
                [yy[e[0]], yy[e[1]]],
                [zz[e[0]], zz[e[1]]],
                color=color,
                alpha=alpha
            )

        # Display item position
        center_x = x + dx/2
        center_y = y + dy/2
        center_z = z + dz/2
        ax.text(center_x, center_y, center_z, f'{x},{y},{z}', color='black', fontsize=8, ha='center', va='center')


# Example Usage
if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # Define bin size and initial slicing
    bin_size = [495, 495, 50]       # [width, height, depth] in mm
    init_slice = [4, 4, 1]            # [slices along x, y, z]
    margin_x = 5                      # Margin in mm along x-axis
    margin_y = 5                      # Margin in mm along y-axis

    generator = TrainsetGenerator(
        bin_size=bin_size,
        init_slice=init_slice,
        margin_x=margin_x,
        margin_y=margin_y
    )

    # Generate trainset with specified number of merges and splits
    trainset = generator.generate_trainset(num_merge=6, num_split=0)

    # Create a unique string for the bin based on size and seed
    str_bin = ''.join(map(str, bin_size)).replace(' ', '').replace('[','').replace(']','').replace(',','')

    # Save the generated trainset to a JSON file
    generator.save_trainset_to_json(
        f"Item_data/exhibition/bin{str_bin}_seed{seed}.json",
        trainset
    )

    # Visualize the generated trainset and save the plot as an image
    generator.visualize_trainset(
        f"Item_data/exhibition/bin{str_bin}_seed{seed}.png",
        save=True
    )
