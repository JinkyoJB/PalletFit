import sys, json, random, os, time
from PyQt5.QtWidgets import QSizePolicy, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QLineEdit, QPushButton, QFrame, QGroupBox, QDialog, QDialogButtonBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import art3d
from planning.item import Item, RotationType


class SaveDialog(QDialog):
    def __init__(self, parent=None):
        super(SaveDialog, self).__init__(parent)
        self.setWindowTitle("Save Results")
        self.layout = QVBoxLayout(self)

        self.label = QLabel("Enter the save path:")
        self.layout.addWidget(self.label)

        self.save_path_input = QLineEdit(self)
        self.save_path_input.setText("planning/data/manual_packing_result/")
        self.layout.addWidget(self.save_path_input)

        self.participant_label = QLabel("Participant:")
        self.layout.addWidget(self.participant_label)

        self.participant_input = QLineEdit(self)
        self.layout.addWidget(self.participant_input)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_save_path(self):
        return self.save_path_input.text()

    def get_participant(self):
        return self.participant_input.text()

class BinPackingSimulation(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        self.reset_ui()

    def reset_ui(self):
        self.queue_offset = 0
        self.target_item_index = 0
        self.target_item = None
        self.pivot = [0, 0, 0]
        self.items = []
        self.items_in_bin = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_rendering)
        self.packing_timer = QTimer()
        self.packing_timer.timeout.connect(self.update_packing_time)
        self.packing_time = 0
        self.create_item_layouts()
        self.update_current_pivot_label()
        self.json_status_label.setText('')
        self.volume_ratio_label.setText('Volume ratio: 0%')
        self.packing_time_label.setText('Packing time: 0 s')
        self.save_path_input.setText('planning/data/manual_packing_result/')
        self.save_group.hide()
        self.ax.clear()
        self.canvas.draw()

    def initUI(self):
        self.setFocusPolicy(Qt.StrongFocus)
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Top layout
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)
        
        # 3D plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas.setFixedSize(600, 600)
        top_layout.addWidget(self.canvas)
        
        # Right layout
        right_layout = QVBoxLayout()
        top_layout.addLayout(right_layout)
        
        # Bin definition group
        bin_group = QGroupBox('Bin Definition')
        bin_layout = QVBoxLayout()
        bin_group.setLayout(bin_layout)
        bin_group.setFixedHeight(100)
        right_layout.addWidget(bin_group)
        
        bin_values_layout = QHBoxLayout()
        bin_values_layout.setSpacing(5)
        bin_layout.addLayout(bin_values_layout)
        
        self.w_spinbox = QDoubleSpinBox()
        self.w_spinbox.setRange(0.0, 10.0)
        self.w_spinbox.setSingleStep(0.1)
        self.w_spinbox.setValue(1.1)
        self.w_spinbox.setValue(1.1)
        bin_values_layout.addWidget(QLabel('Width (w) [m]:'))
        bin_values_layout.addWidget(self.w_spinbox)
        
        self.h_spinbox = QDoubleSpinBox()
        self.h_spinbox.setRange(0.0, 10.0)
        self.h_spinbox.setSingleStep(0.1)
        self.h_spinbox.setValue(1.1)
        self.h_spinbox.setValue(1.1)
        bin_values_layout.addWidget(QLabel('Height (h) [m]:'))
        bin_values_layout.addWidget(self.h_spinbox)
        
        self.d_spinbox = QDoubleSpinBox()
        self.d_spinbox.setRange(0.0, 10.0)
        self.d_spinbox.setSingleStep(0.1)
        self.d_spinbox.setValue(1.8)
        self.d_spinbox.setValue(1.8)
        bin_values_layout.addWidget(QLabel('Depth (d) [m]:'))
        bin_values_layout.addWidget(self.d_spinbox)
        
        self.select_bin_button = QPushButton('Select')
        self.select_bin_button.clicked.connect(self.create_bin)
        bin_values_layout.addWidget(self.select_bin_button)
        
        # Item definition group
        item_group = QGroupBox('Item Definition')
        item_layout = QVBoxLayout()
        item_layout.setSpacing(10)
        item_group.setLayout(item_layout)
        right_layout.addWidget(item_group)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(5)
        item_layout.addLayout(bottom_layout)
        
        self.file_input = QLineEdit()
        self.file_input.setText('planning/data/Item_data/exhibition/debuging_skt2.json')
        bottom_layout.addWidget(self.file_input)
        
        self.select_button = QPushButton('Select')
        self.select_button.clicked.connect(self.load_problem)
        bottom_layout.addWidget(self.select_button)
        
        # JSON load status label
        self.json_status_label = QLabel('')
        item_layout.addWidget(self.json_status_label)
        
        # Number of items before packing
        items_before_packing_layout = QHBoxLayout()
        items_before_packing_layout.setSpacing(5) 
        self.items_label = QLabel('Packing 전에 알 수 있는 아이템 개수:')
        items_before_packing_layout.addWidget(self.items_label)
        
        self.items_spinbox = QSpinBox()
        self.items_spinbox.setRange(1, 4)
        items_before_packing_layout.addWidget(self.items_spinbox)
        
        self.items_select_button = QPushButton('Select')
        self.items_select_button.clicked.connect(self.create_item_layouts)
        items_before_packing_layout.addWidget(self.items_select_button)
        
        item_layout.addLayout(items_before_packing_layout)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.show_prev_page)
        self.next_btn.clicked.connect(self.show_next_page)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        item_layout.addLayout(nav_layout)

                
        # Container for item layouts
        self.item_layout_container = QHBoxLayout()
        item_layout.addLayout(self.item_layout_container)
        
        # Packing layout
        packing_group = QGroupBox('Packing')
        packing_layout = QVBoxLayout()
        packing_group.setLayout(packing_layout)
        right_layout.addWidget(packing_group)

        # Description label
        description_label1 = QLabel('x이동: 좌우 버튼 또는 키보드 좌우버튼')
        packing_layout.addWidget(description_label1)
        description_label2 = QLabel('y이동: 대각선버튼 또는 shift+키보드좌우버튼')
        packing_layout.addWidget(description_label2)
        description_label3 = QLabel('z이동: 위아래버튼 또는 키보드 위아래 버튼')
        packing_layout.addWidget(description_label3)

        # Packing Start button
        self.packing_start_button = QPushButton('Packing Start')
        self.packing_start_button.setFixedSize(150, 50)
        self.packing_start_button.clicked.connect(self.start_packing)
        packing_layout.addWidget(self.packing_start_button, alignment=Qt.AlignCenter)
        
        # Directional buttons layout
        direction_group = QGroupBox('Directional Controls')
        direction_buttons_layout = QVBoxLayout()
        direction_group.setLayout(direction_buttons_layout)
        direction_group.setFixedWidth(330)

        up_button_layout = QHBoxLayout()
        self.up_left_button = QPushButton("↖")
        self.up_left_button.setFixedSize(QSize(100, 50))
        self.up_left_button.clicked.connect(lambda: self.move_item('up_left'))
        self.up_button = QPushButton("↑")
        self.up_button.setFixedSize(QSize(100, 50))
        self.up_button.clicked.connect(lambda: self.move_item('up'))
        self.up_right_button = QPushButton("↗")
        self.up_right_button.setFixedSize(QSize(100, 50))
        self.up_right_button.clicked.connect(lambda: self.move_item('up_right'))
        up_button_layout.addWidget(self.up_left_button)
        up_button_layout.addStretch(1)
        up_button_layout.addWidget(self.up_button)
        up_button_layout.addStretch(1)
        up_button_layout.addWidget(self.up_right_button)

        down_left_right_buttons_layout = QHBoxLayout()
        self.left_button = QPushButton("←")
        self.left_button.setFixedSize(QSize(100, 50))
        self.left_button.clicked.connect(lambda: self.move_item('left'))
        self.down_button = QPushButton("↓")
        self.down_button.setFixedSize(QSize(100, 50))
        self.down_button.clicked.connect(lambda: self.move_item('down'))
        self.right_button = QPushButton("→")
        self.right_button.setFixedSize(QSize(100, 50))
        self.right_button.clicked.connect(lambda: self.move_item('right'))
        down_left_right_buttons_layout.addWidget(self.left_button)
        down_left_right_buttons_layout.addWidget(self.down_button)
        down_left_right_buttons_layout.addWidget(self.right_button)

        direction_buttons_layout.addLayout(up_button_layout)
        direction_buttons_layout.addLayout(down_left_right_buttons_layout)

        pivot_group = QGroupBox('Pivot Controls')
        pivot_controls_layout = QVBoxLayout()
        pivot_group.setLayout(pivot_controls_layout)
        pivot_group.setFixedWidth(330)
        pivot_controls_layout.setSpacing(5)

        pivot_label = QLabel('Pivot:')
        self.pivot_x_spinbox = QDoubleSpinBox()
        self.pivot_x_spinbox.setRange(-10000.0, 10000.0)
        self.pivot_x_spinbox.setSingleStep(100.0)
        self.pivot_x_spinbox.setValue(0.0)
        
        self.pivot_y_spinbox = QDoubleSpinBox()
        self.pivot_y_spinbox.setRange(-10000.0, 10000.0)
        self.pivot_y_spinbox.setSingleStep(100.0)
        self.pivot_y_spinbox.setValue(0.0)
        
        self.pivot_z_spinbox = QDoubleSpinBox()
        self.pivot_z_spinbox.setRange(-10000.0, 10000.0)
        self.pivot_z_spinbox.setSingleStep(100.0)
        self.pivot_z_spinbox.setValue(0.0)
        
        self.try_button = QPushButton('Try')
        self.try_button.clicked.connect(self.update_pivot)
        
        pivot_controls_layout.addWidget(pivot_label)
        pivot_controls_layout.addWidget(QLabel('X:'))
        pivot_controls_layout.addWidget(self.pivot_x_spinbox)
        pivot_controls_layout.addWidget(QLabel('Y:'))
        pivot_controls_layout.addWidget(self.pivot_y_spinbox)
        pivot_controls_layout.addWidget(QLabel('Z:'))
        pivot_controls_layout.addWidget(self.pivot_z_spinbox)
        pivot_controls_layout.addWidget(self.try_button)
        
        # Rotation button
        self.rotation_button = QPushButton('Rotate')
        self.rotation_button.clicked.connect(self.rotate_item)
        pivot_controls_layout.addWidget(self.rotation_button)
        
        # Horizontal layout for direction and pivot controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(direction_group)
        controls_layout.addWidget(pivot_group)
        packing_layout.addLayout(controls_layout)

        # Current pivot label and item packing button layout
        pivot_and_packing_layout = QHBoxLayout()
        self.current_pivot_label = QLabel('Current Pivot: [0, 0, 0]')
        pivot_and_packing_layout.addWidget(self.current_pivot_label)

        self.packing_button = QPushButton('Packing')
        self.packing_button.setFixedSize(150, 50)
        self.packing_button.setStyleSheet("padding: 10px;")
        self.packing_button.clicked.connect(self.pack_item)
        pivot_and_packing_layout.addWidget(self.packing_button, alignment=Qt.AlignCenter)
        
        packing_layout.addLayout(pivot_and_packing_layout)
        
        
        self.items = []
        self.item_vbox_list = []
        self.item_canvases = []
        
        # Result layout
        result_group = QGroupBox('Result')
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        self.volume_ratio_label = QLabel('Volume ratio: 0%')
        result_layout.addWidget(self.volume_ratio_label)

        self.packing_time_label = QLabel('Packing time: 0 s')
        result_layout.addWidget(self.packing_time_label)

        self.save_group = QGroupBox('Save Results')
        save_layout = QVBoxLayout()
        self.save_group.setLayout(save_layout)
        self.save_group.hide()  # Hide by default
        result_layout.addWidget(self.save_group)

        save_layout.addWidget(QLabel('Save file path:'))
        self.save_path_input = QLineEdit()
        self.save_path_input.setText('planning/data/manual_packing_result')
        save_layout.addWidget(self.save_path_input)

        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save_results)
        save_layout.addWidget(save_button)

        confirm_button = QPushButton('Confirm')
        confirm_button.clicked.connect(self.reset_ui)
        save_layout.addWidget(confirm_button)

        # New buttons for restarting and saving current results
        new_buttons_layout = QHBoxLayout()
        self.restart_button = QPushButton('Restart')
        self.restart_button.clicked.connect(self.reset_ui)
        new_buttons_layout.addWidget(self.restart_button)

        self.save_current_button = QPushButton('Save Current')
        self.save_current_button.clicked.connect(self.open_save_dialog)
        new_buttons_layout.addWidget(self.save_current_button)

        result_layout.addLayout(new_buttons_layout)

        self.setWindowTitle('Bin Packing Simulation')
        self.show()

    def _plotCube(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, linewidth=1, text="", fontsize=15, alpha=0.5):
        """ Auxiliary function to plot a cube. """
        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]

        kwargs = {'alpha': 1, 'color': color, 'linewidth': linewidth}
        if mode == 1:
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
        else:
            p = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)

            if text != "":
                ax.text((x + dx / 2), (y + dy / 2), (z + dz / 2), str(text), color='black', fontsize=fontsize, ha='center', va='center')

            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x + dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y + dy, zdir="y")

    def create_bin(self):
        self.bin_w = self.w_spinbox.value() * 1000  # Convert meters to millimeters
        self.bin_h = self.h_spinbox.value() * 1000  # Convert meters to millimeters
        self.bin_d = self.d_spinbox.value() * 1000  # Convert meters to millimeters
        
        # Clear previous bin
        self.ax.clear()
        
        # Draw new bin
        self._plotCube(self.ax, 0, 0, 0, self.bin_w, self.bin_h, self.bin_d, color='cyan', mode=1, linewidth=1, alpha=0.5)
        
        self.ax.set_xlabel('X [mm]')
        self.ax.set_ylabel('Y [mm]')
        self.ax.set_zlabel('Z [mm]')
        
        self.ax.set_xlim([0, max(self.bin_w, 1000)])
        self.ax.set_ylim([0, max(self.bin_h, 1000)])
        self.ax.set_zlim([0, max(self.bin_d, 1000)])

        self.ax.set_xticks(range(0, int(max(self.bin_w, 1000)) + 1, 200))
        self.ax.set_yticks(range(0, int(max(self.bin_h, 1000)) + 1, 200))
        self.ax.set_zticks(range(0, int(max(self.bin_d, 1000)) + 1, 200))
        
        self.canvas.draw()
    
    def load_problem(self):
        file_path = self.file_input.text()
        try:
            with open(file_path, 'r') as file:
                raw = json.load(file)

            # ① 리스트 or 딕셔너리 모두 list[dict] 로 평탄화
            raw_list = raw if isinstance(raw, list) else raw.get('items', [])

            # ② dict → Item 객체 변환 + 임의 색상 부여
            self.items = []
            # random.seed(42)
            # random.shuffle(raw_list)

            for dat in raw_list:
                itm = Item(**{
                    **dat,                           # width/height/depth/weight/unit ...
                    "weight": dat.get("weight", 0),  # 없으면 0
                    "unit":   dat.get("unit", "mm")
                })
                # 👉 PyQt 색이 필요하므로 Hex 문자열을 속성으로 붙여 둔다
                itm.color = dat.get("color") or "#{:02X}{:02X}{:02X}".format(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                self.items.append(itm)

            self.json_status_label.setText("JSON 불러오기 성공")
            self.json_status_label.setStyleSheet("color: green")

        except Exception as e:
            print("Failed to load problem:", e)
            self.json_status_label.setText("JSON 불러오기 실패, 주소를 확인하세요")
            self.json_status_label.setStyleSheet("color: red")

    def set_target(self, queue_idx: int):
        """아이템 큐(index 기준)에서 클릭한 아이템을 현재 타깃으로 지정"""
        if queue_idx >= len(self.items):
            return
        self.target_item_index = queue_idx
        self.target_item = self.items[queue_idx]
        self.pivot = [0, 0, 0]           # 피벗 초기화
        self.update_item_canvas()        # 하이라이트 갱신
        self.update_rendering()

    
    def create_item_layouts(self):
        # 레이아웃 비우기
        while self.item_layout_container.count():
            child = self.item_layout_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.item_vbox_list = []

        num_slots = self.items_spinbox.value()

        for slot in range(num_slots):
            global_idx = self.queue_offset + slot          # ❗ 페이지 기준
            if global_idx >= len(self.items):
                break

            itm = self.items[global_idx]
            w, h, d = itm.getDimension()

            # ── 썸네일 위젯 ─────────────────────────────────────────────
            wrapper = QWidget()
            wrapper.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

            vbox = QVBoxLayout(wrapper)
            vbox.setContentsMargins(0, 0, 0, 0)

            lbl_name = QLabel(f'Item {global_idx + 1}: {itm.name}')
            lbl_whd  = QLabel(f'WHD: {w, h, d}')
            vbox.addWidget(lbl_name)
            vbox.addWidget(lbl_whd)

            fig = Figure(figsize=(2, 2))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111, projection='3d')
            self._plotCube(ax, 0, 0, 0, w, h, d,
                        color=itm.color, mode=2, linewidth=1, alpha=0.5)
            for a in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                a([0, 500])
            ax.axis("off")
            canvas.draw()
            vbox.addWidget(canvas)

            wrapper.mousePressEvent = (
                lambda e, idx=global_idx: self.set_target(idx)  # 절대 인덱스
            )

            self.item_vbox_list.append((lbl_name, lbl_whd, canvas, wrapper))
            self.item_layout_container.addWidget(wrapper)


    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.layout():
                self.clear_layout(child.layout())
            if child.widget():
                child.widget().deleteLater()

    def show_prev_page(self):
        if self.queue_offset - self.items_spinbox.value() >= 0:
            self.queue_offset -= self.items_spinbox.value()
            self.create_item_layouts()

    def show_next_page(self):
        if self.queue_offset + self.items_spinbox.value() < len(self.items):
            self.queue_offset += self.items_spinbox.value()
            self.create_item_layouts()

    
    def start_packing(self):
        if self.items:
            self.target_item_index = 0
            self.target_item = self.items[self.target_item_index]
            self.pivot = [0, 0, 0]
            self.timer.start(500)  # Start the timer with 500ms interval
            self.packing_time = 0
            self.packing_timer.start(1000)  # Update every second
            self.update_item_canvas()

    def move_item(self, direction):
        if self.target_item:
            if direction == 'up':
                self.pivot[2] += 100  # Move up along the z-axis
            elif direction == 'down':
                self.pivot[2] -= 100  # Move down along the z-axis
            elif direction == 'left':
                self.pivot[0] -= 100  # Move left along the x-axis
            elif direction == 'right':
                self.pivot[0] += 100  # Move right along the x-axis
            elif direction == 'up_left':
                self.pivot[1] -= 100  # Move y-axis down
            elif direction == 'up_right':
                self.pivot[1] += 100  # Move y-axis up
        self.update_current_pivot_label()

    def update_pivot(self):
        if self.target_item:
            self.pivot[0] = self.pivot_x_spinbox.value()
            self.pivot[1] = self.pivot_y_spinbox.value() 
            self.pivot[2] = self.pivot_z_spinbox.value()
        self.update_current_pivot_label()

    def rectIntersect_preview(self, target: Item, fixed: Item) -> bool:
        """
        target 은 현재 pivot 을 기준으로 한 미리보기 아이템
        fixed  는 이미 bin 에 적재된 아이템
        세 축 모두 겹치면 True
        """
        tw, th, td = target.getDimension()
        fw, fh, fd = fixed.getDimension()

        # target AABB
        tx1, ty1, tz1 = self.pivot
        tx2, ty2, tz2 = tx1 + tw, ty1 + th, tz1 + td

        # fixed AABB
        fx1, fy1, fz1 = fixed.b_position
        fx2, fy2, fz2 = fx1 + fw, fy1 + fh, fz1 + fd

        overlap_x = not (tx2 <= fx1 or fx2 <= tx1)
        overlap_y = not (ty2 <= fy1 or fy2 <= ty1)
        overlap_z = not (tz2 <= fz1 or fz2 <= tz1)

        return overlap_x and overlap_y and overlap_z


    def rectIntersect_preview(self, target: Item, fixed: Item) -> bool:
        """
        target 은 현재 pivot 을 기준으로 한 미리보기 아이템
        fixed  는 이미 bin 에 적재된 아이템
        세 축 모두 겹치면 True
        """
        tw, th, td = target.getDimension()
        fw, fh, fd = fixed.getDimension()

        # target AABB
        tx1, ty1, tz1 = self.pivot
        tx2, ty2, tz2 = tx1 + tw, ty1 + th, tz1 + td

        # fixed AABB
        fx1, fy1, fz1 = fixed.b_position
        fx2, fy2, fz2 = fx1 + fw, fy1 + fh, fz1 + fd

        overlap_x = not (tx2 <= fx1 or fx2 <= tx1)
        overlap_y = not (ty2 <= fy1 or fy2 <= ty1)
        overlap_z = not (tz2 <= fz1 or fz2 <= tz1)

        return overlap_x and overlap_y and overlap_z

    def update_rendering(self):
        if self.target_item:
            self.ax.clear()
            self.create_bin()
            self.create_bin()

            # 3-A. 이미 적재된 아이템 그리기
            for itm in self.items_in_bin:
                x, y, z = itm.b_position
                w,h,d = itm.getDimension()
                self._plotCube(self.ax, x, y, z,
                            w,h,d,
                            color=itm.color, mode=2, linewidth=1, alpha=0.5)

            # 3-B. 충돌 체크
            preview_ok = all(
                not self.rectIntersect_preview(self.target_item, itm)
                for itm in self.items_in_bin
            ) and not self.out_of_bin_bounds(self.target_item)

            edge_color = 'green' if preview_ok else 'red'

            # 3-C. 미리 보기 아이템
            w, h, d = self.target_item.getDimension()
            self._plotCube(self.ax, *self.pivot, w, h, d,
                        color=edge_color, mode=1, linewidth=1, alpha=0.5)

            # 3-A. 이미 적재된 아이템 그리기
            for itm in self.items_in_bin:
                x, y, z = itm.b_position
                w,h,d = itm.getDimension()
                self._plotCube(self.ax, x, y, z,
                            w,h,d,
                            color=itm.color, mode=2, linewidth=1, alpha=0.5)

            # 3-B. 충돌 체크
            preview_ok = all(
                not self.rectIntersect_preview(self.target_item, itm)
                for itm in self.items_in_bin
            ) and not self.out_of_bin_bounds(self.target_item)

            edge_color = 'green' if preview_ok else 'red'

            # 3-C. 미리 보기 아이템
            w, h, d = self.target_item.getDimension()
            self._plotCube(self.ax, *self.pivot, w, h, d,
                        color=edge_color, mode=1, linewidth=1, alpha=0.5)

            self.canvas.draw()
            self.update_current_pivot_label()



    def out_of_bin_bounds(self, itm: Item):
        return (
            self.pivot[0] + itm.width  > self.bin_w or
            self.pivot[1] + itm.height > self.bin_h or
            self.pivot[2] + itm.depth  > self.bin_d or
            any(c < 0 for c in self.pivot)
        )
        
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.move_item('up')
        elif event.key() == Qt.Key_Down:
            self.move_item('down')
        elif event.key() == Qt.Key_Left:
            if event.modifiers() == Qt.ShiftModifier:
                self.move_item('up_left')
            else:
                self.move_item('left')
        elif event.key() == Qt.Key_Right:
            if event.modifiers() == Qt.ShiftModifier:
                self.move_item('up_right')
            else:
                self.move_item('right')

    def pack_item(self):
        if (not self.target_item or
            any(self.rectIntersect_preview(self.target_item, itm)
                for itm in self.items_in_bin) or
            self.out_of_bin_bounds(self.target_item)):
            return

        # 중력 적용·적재
        self.apply_gravity_to_target_item()
        self.target_item.b_position = self.pivot.copy()
        self.items_in_bin.append(self.target_item)

        # 큐에서 제거
        self.items.remove(self.target_item)

        # 🔹 페이지 보정
        if self.queue_offset >= len(self.items):
            self.queue_offset = max(0, len(self.items) - self.items_spinbox.value())

        # 🔹 다음 타깃 선택 (페이지 첫 칸)
        if self.items:
            self.target_item_index = self.queue_offset
            self.target_item = self.items[self.target_item_index]
            self.pivot = [0, 0, 0]
        else:             # 아이템이 다 없어졌다면
            self.target_item_index = 0
            self.target_item = None

        # UI 갱신
        self.create_item_layouts()
        self.update_rendering()
        self.volume_ratio_label.setText(
            f"Volume ratio: {self.calculate_volume_ratio():.2f}%")


    def update_item_canvas(self):
        num_slots = self.items_spinbox.value()

        for slot in range(num_slots):
            global_idx = self.queue_offset + slot
            if global_idx >= len(self.items):
                break

            itm = self.items[global_idx]
            lbl_name, lbl_whd, canvas, wrapper = self.item_vbox_list[slot]

            lbl_name.setText(f'Item {global_idx+1}: {itm.name}')
            w, h, d = itm.getDimension()
            lbl_whd.setText(f'WHD: {w, h, d}')

            # 미리보기 그림 갱신
            ax = canvas.figure.axes[0]
            ax.clear()
            self._plotCube(ax, 0, 0, 0, w, h, d,
                        color=itm.color, mode=2, linewidth=1, alpha=0.5)
            for f in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                f([0, 500])
            ax.axis("off")
            canvas.draw()

            wrapper.setStyleSheet(
                "border:2px solid red;" if itm is self.target_item else ""
            )

    def apply_gravity_to_target_item(self):
        if not self.target_item:
            return

        max_z = 0
        tw, th, _ = self.target_item.getDimension()

        for itm in self.items_in_bin:
            fw, fh, fd = itm.getDimension()
            ix = (self.pivot[0] < itm.b_position[0] + fw) and \
                (self.pivot[0] + tw > itm.b_position[0])
            iy = (self.pivot[1] < itm.b_position[1] + fh) and \
                (self.pivot[1] + th > itm.b_position[1])
            if ix and iy:
                max_z = max(max_z, itm.b_position[2] + fd)
        if not self.target_item:
            return

        max_z = 0
        tw, th, _ = self.target_item.getDimension()

        for itm in self.items_in_bin:
            fw, fh, fd = itm.getDimension()
            ix = (self.pivot[0] < itm.b_position[0] + fw) and \
                (self.pivot[0] + tw > itm.b_position[0])
            iy = (self.pivot[1] < itm.b_position[1] + fh) and \
                (self.pivot[1] + th > itm.b_position[1])
            if ix and iy:
                max_z = max(max_z, itm.b_position[2] + fd)

        self.pivot[2] = max_z
        self.update_rendering()
        self.pivot[2] = max_z
        self.update_rendering()


    def update_current_pivot_label(self):
        self.current_pivot_label.setText(f'Current Pivot: {self.pivot}')

    def rotate_item(self):
        if self.target_item:
            #self.target_item.rotation_quat가 RotationType.RT_WHD 또는 RotationType.RT_HWD로 토글
            if self.target_item.rotation_quat == RotationType.RT_WHD:
                self.target_item.rotation_quat = RotationType.RT_HWD
            else:
                self.target_item.rotation_quat = RotationType.RT_WHD
            self.target_item._face_info.clear()     # 면 캐시 초기화
            #self.target_item.rotation_quat가 RotationType.RT_WHD 또는 RotationType.RT_HWD로 토글
            if self.target_item.rotation_quat == RotationType.RT_WHD:
                self.target_item.rotation_quat = RotationType.RT_HWD
            else:
                self.target_item.rotation_quat = RotationType.RT_WHD
            self.target_item._face_info.clear()     # 면 캐시 초기화
            self.update_rendering()

    
    def calculate_volume_ratio(self):
        bin_vol = self.bin_w * self.bin_h * self.bin_d
        item_vol = sum(itm.volume for itm in self.items_in_bin)
        return item_vol / bin_vol * 100
        bin_vol = self.bin_w * self.bin_h * self.bin_d
        item_vol = sum(itm.volume for itm in self.items_in_bin)
        return item_vol / bin_vol * 100

    def update_packing_time(self):
        self.packing_time += 1
        self.packing_time_label.setText(f'Packing time: {self.packing_time} s')

    def save_results(self):
        file_path = self.save_path_input.text()
        self.ensure_directory_exists(file_path)
        time.sleep(0.1)  # Add a delay to ensure the directory is created before saving the file
        if file_path:
            self.canvas.figure.savefig(file_path + 'canvas_image.png')  # Save canvas as image
            with open(file_path, 'w') as f:
                f.write(f"Number of items: {len(self.items_in_bin)}\n")
                f.write(f"Volume ratio: {self.volume_ratio_label.text()}\n")
                f.write(f"Packing time: {self.packing_time_label.text()}\n")
                f.write("Bin contents:\n")
                for item in self.items_in_bin:
                    f.write(f"{item}\n")

    def open_save_dialog(self):
        dialog = SaveDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            file_path = dialog.get_save_path()
            participant = dialog.get_participant()
            self.ensure_directory_exists(file_path)
            self.save_current_results(file_path, participant)
            self.packing_timer.stop()
            self.reset_ui()

    def ensure_directory_exists(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_current_results(self, file_path, participant):
        if file_path:
            time_stamp = time.strftime('%Y%m%d_%H%M%S')
            image_path = file_path + time_stamp + '_canvas.png'
            self.canvas.figure.savefig(image_path)  # Save canvas as image
            with open(file_path + time_stamp, 'w') as f:
                f.write(f"Participant: {participant}\n")
                f.write(f"case: {self.file_input.text()}\n")
                f.write(f"Current number of items: {len(self.items_in_bin)}\n")
                f.write(f"Current volume ratio: {self.volume_ratio_label.text()}\n")
                f.write(f"Packing time: {self.packing_time} s\n")
                f.write("Current bin contents:\n")
                for item in self.items_in_bin:
                    f.write(f"{item}\n")





if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BinPackingSimulation()
    sys.exit(app.exec_())
