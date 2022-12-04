import sys, os

from PyQt5.QtWidgets import * # QApplication, QMainWindow, QWidget, QMenu, etc...
from PyQt5.QtCore import *
import qtawesome as qta

from ui_qt.ui_main import Ui_MainWindow
from functions import WorkerThread, Functions
from values import dict_liquid, feature_liquid, dict_methods, CaWeMo
from lib.loglo import init_log

lo = init_log(__file__, save_log=True, level=os.environ.get("LOG_LEVEL", "0"))

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        icon = qta.icon("ph.test-tube-fill")
        self.setWindowIcon(icon)

        self.line_edits_liquid = []
        self.radio_methods = []
        try:
            self.str_liquid = list(dict_liquid.keys())[0]
        except:
            self.str_liquid = "unknown"
        self.idx_experiment = 1
        self.method_name = 'LR'

        self.setDefaultValues()
        self.setEvents()

        # # right click menu
        # self.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.customContextMenuRequested.connect(self.right_menu)

        # # set the window corner to be rounded
        # self.set_marui()
    
    def setDefaultValues(self):
        # self.setWindowTitle('Title')
        ui = self.ui

        for name in dict_liquid.keys():
            ui.comboBox_type_liquid.addItem(name)
        # ui.comboBox_type_liquid.addItem("toluene")
        # ui.comboBox_type_liquid.addItem("heptane")
        # ui.comboBox_type_liquid.addItem("butanol")
        # ui.comboBox_type_liquid.addItem("ethyl acetate")
        # ui.comboBox_type_liquid.addItem("unknown")

        # positions = [(i, j) for i in range(5) for j in range(4)]
        for col, name in enumerate(feature_liquid):
            label = QLabel(name)
            ui.gridLayout_liquid_vals.addWidget(label, 0, col)

            line_edit = QLineEdit()
            ui.gridLayout_liquid_vals.addWidget(line_edit, 1, col)
            self.line_edits_liquid.append(line_edit)

        self.set_liquid_value()

        # --
        ui.comboBox_experiment_2.addItem("1. Maximum Input Flow Rate Prediction")
        ui.comboBox_experiment_2.addItem("2. Prediction of Droplet Separation")
        # --
        self.ui.lineEdit_trainPath.setText(r"./training_data/exp1.csv")
        self.ui.lineEdit_testPath.setText(r"./test_data/exp1.csv")
        # --
        ui.comboBox_experiment.addItem("1. Maximum Input Flow Rate Prediction")
        ui.comboBox_experiment.addItem("2. Prediction of Droplet Separation")
        # --
        ui.lineEdit_flowrate.setText("0")
        ui.lineEdit_pitch.setText("0")
        ui.lineEdit_total_flow_rate.setText("0")
        # --
        ui.comboBox_ratio.addItem("1:1")
        ui.comboBox_ratio.addItem("2:3")
        ui.comboBox_ratio.addItem("1:3")
        ui.comboBox_L.addItem("3")
        ui.comboBox_L.addItem("7")
        ui.comboBox_L.addItem("14")
        # --
        self.radioGroup = QButtonGroup()
        self.radioGroup.buttonClicked.connect(self.check_method)
        self.update_layout()
        # --
        # self.ui.textEdit_res.append("")
        
        # --
        self.statusBar().showMessage('')
    
    def update_layout(self):
        ui = self.ui
        if self.idx_experiment == 1:
            ui.lineEdit_flowrate.setEnabled(True)
            ui.comboBox_ratio.setEnabled(False)
            ui.lineEdit_total_flow_rate.setEnabled(False)
        elif self.idx_experiment == 2:
            ui.lineEdit_flowrate.setEnabled(False)
            ui.comboBox_ratio.setEnabled(True)
            ui.lineEdit_total_flow_rate.setEnabled(True)

        methods = dict_methods.get(self.idx_experiment)
        for widget in self.radio_methods:
            ui.gridLayout_method.removeWidget(widget)
            self.radioGroup.removeButton(widget)

        self.radio_methods = []
        for col, name in enumerate(methods):
            radioBtn = QRadioButton()

            ui.gridLayout_method.addWidget(radioBtn, 1, col)
            self.radioGroup.addButton(radioBtn)
            radioBtn.setText(name)
            self.radio_methods.append(radioBtn)
        
        if len(self.radio_methods) > 0:
            assert isinstance(self.radio_methods[0], QRadioButton)
            self.radio_methods[0].click()
        
    def check_method(self):
        for radio in self.radio_methods:
            if radio.isChecked():
                # print("select radio ", radio.text())
                self.method_name = radio.text()
    
    def setEvents(self):
        ui = self.ui
        
        # ui.pushButton.clicked.connect(lambda : Button.btn_pressed(123))
        ui.actionRun.triggered.connect(self.on_execute)
        ui.actionTrain.triggered.connect(self.on_training)
        ui.actionQuit.triggered.connect(app.exit)

        ui.comboBox_type_liquid.currentTextChanged.connect(self.on_liquid_changed)
        ui.comboBox_experiment.currentIndexChanged.connect(self.on_experiment_changed)
        ui.comboBox_experiment_2.currentIndexChanged.connect(self.on_experiment_2_changed)
        ui.pushButton.clicked.connect(self.on_execute)

        ui.pushButton_setTrainPath.clicked.connect(self.on_set_train_path)
        ui.pushButton_setTestPath.clicked.connect(self.on_set_test_path)
        ui.pushButton_training.clicked.connect(self.on_training)
        
    def on_set_train_path(self):
        path = QFileDialog.getOpenFileName(self, 'Choose training data file', './', "csv file (*.csv)")
        if path[0] != "":
            self.ui.lineEdit_trainPath.setText(path[0])
    
    def on_set_test_path(self):
        path = QFileDialog.getOpenFileName(self, 'Choose testing data file', './', "csv file (*.csv)")
        if path[0] != "":
            self.ui.lineEdit_testPath.setText(path[0])

    def on_execute(self):
        self.do_execute()
    
    def do_execute(self):
        self.statusBar().showMessage("")
        ui = self.ui
        exp_num = self.idx_experiment  # 1 or 2
        method = self.method_name
        liquid_name = self.str_liquid
        
        if exp_num == 1:
            try:
                flow_rate = float(ui.lineEdit_flowrate.text())
            except Exception as ex:
                msg_err = "flow_rate not valid! "
                lo.error( f"{msg_err} [{ui.lineEdit_flowrate.text()}]")
                self.statusBar().showMessage(msg_err)
                return
        elif exp_num == 2:
            try:
                vals_ratio = ui.comboBox_ratio.currentText().split(":")
                ratio = float(vals_ratio[1]) / float(vals_ratio[0])
                total_flow_rate = float(ui.lineEdit_total_flow_rate.text())
            except Exception as ex:
                msg_err = "ratio or total_flow_rate not valid! ratio must be 'number:number' e.g. '1:1'"
                lo.error(f"{msg_err} [{ui.comboBox_ratio.currentText()}, {ui.lineEdit_total_flow_rate.text()}]")
                self.statusBar().showMessage(msg_err)
                return

        try:
            custom_L = int(ui.comboBox_L.currentText())
            pitch = float(ui.lineEdit_pitch.text())
        except Exception as ex:
            msg_err = "pitch or L not valid! "
            lo.error(f"{msg_err} [{ui.lineEdit_pitch.text()} {ui.comboBox_L.currentText()}]")
            self.statusBar().showMessage(msg_err)
            return

        try:
            ca = CaWeMo.get_ca(liquid_name)
            polarity = dict_liquid.get(liquid_name).polarityIndex
            bo = CaWeMo.get_bo(liquid_name)
            mo = CaWeMo.get_mo(liquid_name)
        except Exception as ex:
            lo.error("ca, bo, mo, polarity of liquid are not valid!")
            self.statusBar().showMessage("ca, bo, mo, polarity of liquid are not valid!")
            return

        
        if exp_num == 1:
            # ['ca', 'polarity', 'bo', 'mo', 'org.flowrate', 'pitch', 'L']
            # arr_custom_data = [0.0457473, 2.4, 0.0346, 4.34e-10, 200, 0.245, 7]
            arr_custom_data = [ca, polarity, bo, mo, flow_rate, pitch, custom_L]
        elif exp_num == 2:
            # ['ratio', 'total_flow_rate', 'polarity', 'ca', 'bo', 'mo', 'pitch', 'L']
            # arr_custom_data = [1, 800, 4, 0.527922, 0.0495, 6.2e-10, 0.245, 7]
            arr_custom_data = [ratio, total_flow_rate, polarity, ca, bo, mo, pitch, custom_L]
        else:
            msg_err = "not a valid experiment!"
            lo.error(msg_err)
            self.statusBar().showMessage(msg_err)
            return

        try:
            res = Functions.execute(exp_num, method, arr_custom_data)
            lo.info(f"{res}")
            ui.textEdit_res.append(res)
            ui.textEdit_res.verticalScrollBar().setValue(ui.textEdit_res.verticalScrollBar().maximum())
        except Exception as ex:
            lo.warn(f"calculation failed. {ex}")
            self.statusBar().showMessage('calculation failed.')
            return
        self.statusBar().showMessage('calculate successfully!')

    def on_training(self):
        self.do_train()

    def do_train(self):
        self.statusBar().showMessage('Now training ... please wait')

        idx_exp = self.ui.comboBox_experiment_2.currentIndex()
        if idx_exp not in [0, 1]:
            lo.warn("not a valid experiment")
            return
        exp_num = idx_exp + 1
        
        path_csv_train = self.ui.lineEdit_trainPath.text()
        path_csv_test = self.ui.lineEdit_testPath.text()
        
        try:
            self.ui.pushButton_training.setDisabled(True)
            self.work = WorkerThread(exp_num, path_csv_train, path_csv_test)
            self.work.start()
            self.work.finished.connect(self.finish_train)
            
        except Exception as ex:
            lo.error(f"{ex}")
        
    def finish_train(self, res_code):
        self.ui.pushButton_training.setDisabled(False)
        if res_code == 0:
            self.statusBar().showMessage('Training successfully.')
        else:
            self.statusBar().showMessage('[Warn] Training failed!')

    def on_liquid_changed(self, value):
        self.str_liquid = value
        self.set_liquid_value()

    def on_experiment_changed(self, idx):
        self.idx_experiment = idx + 1
        self.update_layout()
    
    def on_experiment_2_changed(self, idx):
        if idx == 0:
            self.ui.lineEdit_trainPath.setText(r"./training_data/exp1.csv")
            self.ui.lineEdit_testPath.setText(r"./test_data/exp1.csv")
        elif idx == 1:
            self.ui.lineEdit_trainPath.setText(r"./training_data/exp2.csv")
            self.ui.lineEdit_testPath.setText(r"./test_data/exp2.csv")

    def set_liquid_value(self):
        arr_vals = [0] * 5
        try:
            arr_vals = dict_liquid.get(self.str_liquid).arr
        except Exception as ex:
            lo.error(f"{ex}")

        for line_edit, val in zip(self.line_edits_liquid, arr_vals):
            line_edit.setText(str(val))

    # def set_marui(self):
    #     self.WIDTH = 300
    #     self.HEIGHT = 300
    #     self.resize(self.WIDTH, self.HEIGHT)
    #     # Initial
    #     # self.setWindowFlag(Qt.FramelessWindowHint)
    #     self.setAttribute(Qt.WA_TranslucentBackground)
    #     self.setWindowOpacity(0.6)
    #     # Widget
    #     self.centralwidget = QWidget(self)
    #     self.centralwidget.resize(self.WIDTH, self.HEIGHT)
    #     radius = 30
    #     self.centralwidget.setStyleSheet(
    #         """
    #         background:rgb(255, 255, 255);
    #         border-top-left-radius:{0}px;
    #         border-bottom-left-radius:{0}px;
    #         border-top-right-radius:{0}px;
    #         border-bottom-right-radius:{0}px;
    #         """.format(radius)
    #     )
    
    ## move window with dragging
    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         self.moveFlag = True
    #         self.movePosition = event.globalPos() - self.pos()
    #         self.setCursor(QCursor(Qt.OpenHandCursor))
    #         event.accept()
    # def mouseMoveEvent(self, event):
    #     if Qt.LeftButton and self.moveFlag:
    #         self.move(event.globalPos() - self.movePosition)
    #         event.accept()
    # def mouseReleaseEvent(self, event):
    #     self.moveFlag = False
    #     self.setCursor(Qt.CrossCursor)

    # def right_menu(self, pos):
    #     menu = QMenu()

    #     # Add menu options
    #     hello_option = menu.addAction('Hello World')
    #     goodbye_option = menu.addAction('GoodBye')
    #     exit_option = menu.addAction('Exit')

    #     # Menu option events
    #     hello_option.triggered.connect(lambda: print('Hello World'))
    #     goodbye_option.triggered.connect(lambda: print('Goodbye'))
    #     exit_option.triggered.connect(lambda: exit())

    #     # Position
    #     menu.exec_(self.mapToGlobal(pos))
    
    def mousePressEvent(self, event):
        pass
        # if event.button() == Qt.LeftButton:
        #     print('Click left btn')

        # if event.button() == Qt.RightButton:
        #     print('Click right btn')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    
    window.show()
    sys.exit(app.exec_())
    