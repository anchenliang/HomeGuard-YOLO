import cv2
import time
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
from datetime import datetime, timedelta
import logging
from queue import Queue
import os
from pathlib import Path

# 创建必要的目录
def create_directories():
    """创建必要的目录"""
    directories = ['pic', 'log']
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {dir_name}")

create_directories()

# 配置日志
log_file = Path('log') / 'security_monitor.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

class SecurityMonitor:
    def __init__(self, email_config, camera_index=0, model_path='yolov8n.pt'):
        """
        初始化监控系统
        
        Args:
            email_config: 邮箱配置字典
            camera_index: 摄像头索引（默认0）
            model_path: YOLOv8模型路径
        """
        self.email_config = email_config
        self.camera_index = camera_index
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=5)  # 报警邮件队列
        self.self_check_queue = Queue(maxsize=5)  # 自检邮件队列
        self.display_queue = Queue(maxsize=5)  # 用于显示画面的队列
        self.is_running = False
        self.pic_dir = Path('pic')
        
        # 系统状态统计
        self.system_start_time = datetime.now()
        self.last_check_time = datetime.now()
        self.check_interval = 12 * 60 * 60  # 12小时（秒）
        self.initial_check_done = False
        self.person_detection_count = 0
        self.alert_email_sent = 0
        self.check_email_sent = 0
        self.failed_email_count = 0
        self.last_frame = None
        
        # 加载YOLOv8模型
        logger.info(f"加载YOLOv8模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        # 邮件发送配置
        self.smtp_server = email_config.get('smtp_server', 'smtp.qq.com')
        self.smtp_port = email_config.get('smtp_port', 465)
        self.sender_email = email_config['sender_email']
        self.sender_password = email_config['sender_password']
        self.receiver_emails = email_config['receiver_emails']  # 修改为接收多个邮箱
        
        # 检测配置
        self.person_class_id = 0  # YOLO中人的类别ID
        self.confidence_threshold = 0.5
        self.last_alert_email_time = 0
        self.alert_email_cooldown = 60  # 报警邮件冷却时间（秒）
        
    def capture_frames(self):
        """捕获摄像头帧"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            # 尝试其他摄像头索引
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    logger.info(f"找到摄像头 {i}")
                    self.camera_index = i
                    break
            if not cap.isOpened():
                logger.error("所有摄像头都无法打开")
                return
        
        logger.info(f"开始捕获摄像头画面 (索引: {self.camera_index})...")
        
        # 尝试设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        logger.info(f"摄像头FPS: {fps}")
        
        frame_interval = int(fps)  # 每秒1帧
        frame_count = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头画面")
                time.sleep(1)
                continue
            
            # 保存最后一帧用于自检
            self.last_frame = frame.copy()
            
            # 实时显示画面
            display_frame = cv2.resize(frame, (640, 480))
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示运行时间
            run_time = datetime.now() - self.system_start_time
            run_time_str = str(run_time).split('.')[0]
            cv2.putText(display_frame, f"Run time: {run_time_str}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示检测统计
            cv2.putText(display_frame, f"Detections: {self.person_detection_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示邮件统计
            cv2.putText(display_frame, f"Alerts: {self.alert_email_sent}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示下一个自检时间
            next_check = self.last_check_time + timedelta(seconds=self.check_interval)
            time_left = next_check - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = int(time_left.total_seconds() / 3600)
                mins_left = int((time_left.total_seconds() % 3600) / 60)
                cv2.putText(display_frame, f"Next check: {hours_left}h {mins_left}m", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示画面
            cv2.imshow('Security Monitor', display_frame)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户按下q键，准备退出...")
                self.is_running = False
                break
            
            frame_count += 1
            if frame_count % frame_interval == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                frame_count = 0
            
            time.sleep(0.01)  # 避免CPU占用过高
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("摄像头捕获线程结束")
    
    def detect_objects(self):
        """检测图片中的人"""
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.1)
                    continue
                
                frame = self.frame_queue.get()
                
                # 使用YOLOv8进行检测
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # 检查是否检测到人
                person_detected = False
                detections_info = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls == self.person_class_id and conf >= self.confidence_threshold:
                                person_detected = True
                                # 获取边界框坐标
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections_info.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': conf
                                })
                
                if person_detected:
                    self.person_detection_count += 1
                    logger.info(f"检测到{len(detections_info)}个人 (累计: {self.person_detection_count})")
                    
                    # 在图片上绘制检测结果
                    annotated_frame = results[0].plot()
                    
                    # 保存图片到pic目录
                    timestamp = datetime.now()
                    filename = timestamp.strftime('%Y%m%d_%H%M%S') + '_alert.jpg'
                    save_path = self.pic_dir / filename
                    cv2.imwrite(str(save_path), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    logger.info(f"报警图片已保存到: {save_path}")
                    
                    # 将检测结果和标注后的图片放入报警队列
                    if not self.detection_queue.full():
                        self.detection_queue.put({
                            'type': 'alert',
                            'frame': annotated_frame,
                            'timestamp': timestamp,
                            'detections': detections_info,
                            'save_path': str(save_path)
                        })
                    
                    # 将标注后的帧放入显示队列
                    if not self.display_queue.full():
                        display_frame = cv2.resize(annotated_frame, (640, 480))
                        cv2.putText(display_frame, "PERSON DETECTED!", (10, 210), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.display_queue.put(display_frame)
                
            except Exception as e:
                logger.error(f"检测过程中出错: {e}")
                time.sleep(1)
        
        logger.info("检测线程结束")
    
    def send_alert_email(self, image, timestamp, detections, save_path):
        """发送报警邮件"""
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_alert_email_time < self.alert_email_cooldown:
            wait_time = int(self.alert_email_cooldown - (current_time - self.last_alert_email_time))
            logger.warning(f"报警邮件发送冷却中，请等待{wait_time}秒")
            return False
        
        try:
            logger.info(f"开始准备发送报警邮件...")
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['Subject'] = f'北京出租屋安全警报：检测到人员入侵 - {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
            msg['From'] = self.sender_email
            
            # 多个收件人用逗号分隔
            if isinstance(self.receiver_emails, list):
                msg['To'] = ', '.join(self.receiver_emails)
            else:
                msg['To'] = self.receiver_emails
            
            # 邮件正文
            body = f"""
            <html>
                <body>
                    <h2>北京出租屋安全警报</h2>
                    <p>检测到人员入侵！</p>
                    <p>时间：{timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>检测到人数：{len(detections)}</p>
                    <p>请查看附件图片。</p>
                </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # 添加图片附件
            logger.info(f"读取报警图片文件: {save_path}")
            with open(save_path, 'rb') as f:
                img_data = f.read()
                image_mime = MIMEImage(img_data, name=os.path.basename(save_path))
                msg.attach(image_mime)
            
            # 发送邮件
            logger.info(f"连接SMTP服务器发送报警邮件...")
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=30)
            server.login(self.sender_email, self.sender_password)
            
            # 发送给多个收件人
            if isinstance(self.receiver_emails, list):
                server.send_message(msg, self.sender_email, self.receiver_emails)
            else:
                server.send_message(msg, self.sender_email, [self.receiver_emails])
            
            server.quit()
            
            logger.info(f"报警邮件发送成功: {save_path}")
            self.last_alert_email_time = current_time
            self.alert_email_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"报警邮件发送失败: {type(e).__name__}: {e}")
            self.failed_email_count += 1
            return False
    
    def send_self_check_email(self, image, timestamp, has_person=False, person_count=0):
        """发送环境自检邮件"""
        try:
            logger.info(f"开始准备发送环境自检邮件...")
            
            # 保存图片到pic目录
            filename = timestamp.strftime('%Y%m%d_%H%M%S') + '_check.jpg'
            save_path = self.pic_dir / filename
            cv2.imwrite(str(save_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            logger.info(f"自检图片已保存到: {save_path}")
            
            # 计算运行时间
            run_time = timestamp - self.system_start_time
            run_time_str = str(run_time).split('.')[0]
            
            # 计算距离下次自检时间
            next_check_time = timestamp + timedelta(seconds=self.check_interval)
            next_check_str = next_check_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 创建邮件
            msg = MIMEMultipart()
            
            # 修改邮件标题为"北京出租屋监控定时自检结果"
            if has_person:
                msg['Subject'] = f'北京出租屋监控定时自检结果：检测到人员 - {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
                person_status = f"检测到人员: {person_count}人"
            else:
                msg['Subject'] = f'北京出租屋监控定时自检结果：正常 - {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
                person_status = "未检测到人员"
            
            msg['From'] = self.sender_email
            
            # 多个收件人用逗号分隔
            if isinstance(self.receiver_emails, list):
                msg['To'] = ', '.join(self.receiver_emails)
            else:
                msg['To'] = self.receiver_emails
            
            # 邮件正文
            body = f"""
            <html>
                <body>
                    <h2>北京出租屋监控定时自检报告</h2>
                    <h3>系统状态</h3>
                    <ul>
                        <li>运行时间: {run_time_str}</li>
                        <li>系统启动: {self.system_start_time.strftime("%Y-%m-%d %H:%M:%S")}</li>
                        <li>自检时间: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}</li>
                        <li>下次自检: {next_check_str}</li>
                    </ul>
                    
                    <h3>检测统计</h3>
                    <ul>
                        <li>人员检测: {person_status}</li>
                        <li>累计检测次数: {self.person_detection_count}</li>
                        <li>报警邮件发送: {self.alert_email_sent}次</li>
                        <li>自检邮件发送: {self.check_email_sent}次</li>
                        <li>邮件发送失败: {self.failed_email_count}次</li>
                    </ul>
                    
                    <h3>系统状态</h3>
                    <ul>
                        <li>摄像头: 正常</li>
                        <li>YOLO检测: 正常</li>
                        <li>邮件服务: 正常</li>
                        <li>程序运行: 正常</li>
                    </ul>
                    
                    <p>附件为当前环境截图。</p>
                </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # 添加图片附件
            with open(save_path, 'rb') as f:
                img_data = f.read()
                image_mime = MIMEImage(img_data, name=os.path.basename(save_path))
                msg.attach(image_mime)
            
            # 发送邮件
            logger.info(f"连接SMTP服务器发送自检邮件...")
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=30)
            server.login(self.sender_email, self.sender_password)
            
            # 发送给多个收件人
            if isinstance(self.receiver_emails, list):
                server.send_message(msg, self.sender_email, self.receiver_emails)
            else:
                server.send_message(msg, self.sender_email, [self.receiver_emails])
            
            server.quit()
            
            logger.info(f"环境自检邮件发送成功: {save_path}")
            self.check_email_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"环境自检邮件发送失败: {type(e).__name__}: {e}")
            self.failed_email_count += 1
            return False
    
    def email_sender(self):
        """邮件发送线程"""
        while self.is_running:
            try:
                # 优先处理报警邮件
                if not self.detection_queue.empty():
                    detection_data = self.detection_queue.get()
                    
                    logger.info(f"处理报警邮件，检测时间: {detection_data['timestamp']}")
                    
                    success = self.send_alert_email(
                        detection_data['frame'],
                        detection_data['timestamp'],
                        detection_data['detections'],
                        detection_data['save_path']
                    )
                    
                    if success:
                        logger.info(f"报警邮件发送成功 (累计: {self.alert_email_sent})")
                    else:
                        logger.warning(f"报警邮件发送失败 (累计失败: {self.failed_email_count})")
                
                # 处理自检邮件
                elif not self.self_check_queue.empty():
                    check_data = self.self_check_queue.get()
                    
                    logger.info(f"处理自检邮件，时间: {check_data['timestamp']}")
                    
                    success = self.send_self_check_email(
                        check_data['frame'],
                        check_data['timestamp'],
                        check_data['has_person'],
                        check_data['person_count']
                    )
                    
                    if success:
                        logger.info(f"自检邮件发送成功 (累计: {self.check_email_sent})")
                    else:
                        logger.warning(f"自检邮件发送失败 (累计失败: {self.failed_email_count})")
                
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"邮件发送线程出错: {e}")
                time.sleep(1)
        
        logger.info(f"邮件发送线程结束，报警邮件: {self.alert_email_sent}, 自检邮件: {self.check_email_sent}, 失败: {self.failed_email_count}")
    
    def self_check_scheduler(self):
        """环境自检调度器"""
        # 等待1分钟发送第一次自检邮件
        logger.info("环境自检调度器启动，等待60秒后发送首次自检邮件...")
        time.sleep(60)
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # 如果是首次自检或者已经过了12小时
                if not self.initial_check_done or \
                   (current_time - self.last_check_time).total_seconds() >= self.check_interval:
                    
                    logger.info(f"开始环境自检，当前时间: {current_time}")
                    
                    # 获取当前帧进行检测
                    if self.last_frame is not None:
                        frame = self.last_frame.copy()
                        
                        # 检测当前帧是否有人
                        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                        
                        has_person = False
                        person_count = 0
                        
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    if cls == self.person_class_id and conf >= self.confidence_threshold:
                                        has_person = True
                                        person_count += 1
                        
                        if has_person:
                            # 在图片上绘制检测结果
                            annotated_frame = results[0].plot()
                            logger.info(f"自检时检测到{person_count}个人")
                        else:
                            annotated_frame = frame
                            logger.info("自检时未检测到人")
                        
                        # 将自检任务放入队列
                        if not self.self_check_queue.full():
                            self.self_check_queue.put({
                                'type': 'self_check',
                                'frame': annotated_frame,
                                'timestamp': current_time,
                                'has_person': has_person,
                                'person_count': person_count
                            })
                        
                        self.last_check_time = current_time
                        self.initial_check_done = True
                        
                        logger.info(f"环境自检任务已加入队列，下次自检在12小时后")
                    
                    else:
                        logger.warning("无法获取当前帧进行自检")
                
                # 每小时检查一次是否到自检时间
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"环境自检调度器出错: {e}")
                time.sleep(60)
        
        logger.info("环境自检调度器结束")
    
    def display_frames(self):
        """显示画面线程"""
        while self.is_running:
            try:
                if not self.display_queue.empty():
                    display_frame = self.display_queue.get()
                    cv2.imshow('Security Monitor - Detected', display_frame)
                    # 短暂显示检测结果窗口
                    cv2.waitKey(1000)
                    cv2.destroyWindow('Security Monitor - Detected')
                else:
                    time.sleep(0.05)
                    
            except Exception as e:
                logger.error(f"显示画面线程出错: {e}")
                time.sleep(0.1)
    
    def start(self):
        """启动监控系统"""
        self.is_running = True
        
        # 创建并启动线程
        threads = []
        
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detect_objects, daemon=True)
        email_thread = threading.Thread(target=self.email_sender, daemon=True)
        display_thread = threading.Thread(target=self.display_frames, daemon=True)
        check_thread = threading.Thread(target=self.self_check_scheduler, daemon=True)
        
        threads.extend([capture_thread, detection_thread, email_thread, display_thread, check_thread])
        
        for thread in threads:
            thread.start()
        
        logger.info("监控系统已启动")
        logger.info("显示画面中，按'q'键退出")
        
        try:
            # 等待所有线程结束
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("接收到停止信号")
        finally:
            self.stop()
    
    def stop(self):
        """停止监控系统"""
        logger.info("正在停止监控系统...")
        self.is_running = False
        time.sleep(2)  # 等待线程结束
        cv2.destroyAllWindows()
        logger.info("监控系统已停止")


def main():
    print("=" * 50)
    print("北京出租屋安全监控系统启动")
    print("=" * 50)
    
    # =========== 配置部分 ===========
    email_config = {
        'smtp_server': 'smtp.qq.com',  # QQ邮箱SMTP服务器
        'smtp_port': 465,              # SSL端口
        'sender_email': '你的邮箱',     # 发件人邮箱（推荐使用qq邮箱）
        'sender_password': '你的授权码',  # 授权码
        # 多个收件人邮箱（列表形式）
        'receiver_emails': [
            '收件人1邮箱',
            '收件人2邮箱'
        ]
    }
    
    print("\n跳过邮箱配置测试，直接启动系统...")
    print("注意：程序启动后1分钟发送首次环境自检邮件")
    print("此后每隔12小时发送一次环境自检邮件")
    print("检测到人时立即发送报警邮件（60秒冷却时间）")
    print(f"收件人: {', '.join(email_config['receiver_emails'])}")
    print("\n按回车键继续，或按Ctrl+C退出")
    input()
    
    print("\n正在启动摄像头...")
    
    # 摄像头索引
    camera_index = 0  # 默认摄像头
    
    # YOLOv8模型路径
    model_path = 'yolov8n.pt'
    
    # 如果模型不存在，尝试下载
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，尝试下载...")
        try:
            model = YOLO('yolov8n.pt')
            print("模型下载成功")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请手动下载模型或检查网络连接")
            return
    
    # 创建并启动监控系统
    print("\n启动监控系统...")
    print(f"图片保存目录: {Path('pic').absolute()}")
    print(f"日志文件: {Path('log') / 'security_monitor.log'}")
    print("=" * 50)
    
    monitor = SecurityMonitor(email_config, camera_index, model_path)
    monitor.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束")
        print("按回车键退出...")
        input()