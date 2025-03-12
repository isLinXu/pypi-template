# Python高级设计模式与异步编程指南

本文档提供了Python高级设计模式与异步编程的全面指南，包括常用设计模式的实现和异步编程技术的高级应用。本文整合了多个相关文档的内容，提供了一站式的学习资源。

## 设计模式进阶

### 观察者模式

观察者模式定义了对象间的一对多依赖关系，当一个对象状态改变时，所有依赖它的对象都会收到通知并自动更新。

```python
from abc import ABC, abstractmethod
from typing import List

# 抽象观察者
class Observer(ABC):
    @abstractmethod
    def update(self, message: str) -> None:
        pass

# 具体观察者
class EmailNotifier(Observer):
    def __init__(self, email: str):
        self.email = email
    
    def update(self, message: str) -> None:
        print(f"发送邮件通知到 {self.email}: {message}")

class SMSNotifier(Observer):
    def __init__(self, phone: str):
        self.phone = phone
    
    def update(self, message: str) -> None:
        print(f"发送短信通知到 {self.phone}: {message}")

# 被观察的主题
class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def notify(self, message: str) -> None:
        for observer in self._observers:
            observer.update(message)

# 具体主题
class NewsPublisher(Subject):
    def __init__(self):
        super().__init__()
        self._latest_news = ""
    
    @property
    def latest_news(self) -> str:
        return self._latest_news
    
    @latest_news.setter
    def latest_news(self, news: str) -> None:
        self._latest_news = news
        self.notify(f"新闻更新: {news}")

# 使用示例
publisher = NewsPublisher()

# 添加观察者
email_notifier = EmailNotifier("user@example.com")
publisher.attach(email_notifier)

sms_notifier = SMSNotifier("13800138000")
publisher.attach(sms_notifier)

# 发布新闻，触发通知
publisher.latest_news = "Python 3.11发布，性能提升25%！"

# 输出：
# 发送邮件通知到 user@example.com: 新闻更新: Python 3.11发布，性能提升25%！
# 发送短信通知到 13800138000: 新闻更新: Python 3.11发布，性能提升25%！

# 移除一个观察者
publisher.detach(sms_notifier)

# 再次发布新闻
publisher.latest_news = "Python 3.12将支持更多新特性！"

# 输出：
# 发送邮件通知到 user@example.com: 新闻更新: Python 3.12将支持更多新特性！
```

### 异步观察者模式

```python
import asyncio
from abc import ABC, abstractmethod
from typing import List, Set

# 异步观察者接口
class AsyncObserver(ABC):
    @abstractmethod
    async def update(self, message: str) -> None:
        pass

# 具体异步观察者
class AsyncEmailNotifier(AsyncObserver):
    def __init__(self, email: str):
        self.email = email
    
    async def update(self, message: str) -> None:
        print(f"开始发送邮件到 {self.email}...")
        await asyncio.sleep(1)  # 模拟发送邮件
        print(f"邮件已发送到 {self.email}: {message}")

class AsyncSMSNotifier(AsyncObserver):
    def __init__(self, phone: str):
        self.phone = phone
    
    async def update(self, message: str) -> None:
        print(f"开始发送短信到 {self.phone}...")
        await asyncio.sleep(0.5)  # 模拟发送短信
        print(f"短信已发送到 {self.phone}: {message}")

# 异步主题
class AsyncSubject:
    def __init__(self):
        self._observers: Set[AsyncObserver] = set()
    
    def attach(self, observer: AsyncObserver) -> None:
        self._observers.add(observer)
    
    def detach(self, observer: AsyncObserver) -> None:
        self._observers.discard(observer)
    
    async def notify(self, message: str) -> None:
        # 并发通知所有观察者
        await asyncio.gather(*[observer.update(message) for observer in self._observers])

# 具体异步主题
class AsyncNewsPublisher(AsyncSubject):
    def __init__(self):
        super().__init__()
        self._latest_news = ""
    
    @property
    def latest_news(self) -> str:
        return self._latest_news
    
    async def publish_news(self, news: str) -> None:
        self._latest_news = news
        print(f"发布新闻: {news}")
        await self.notify(f"新闻更新: {news}")

# 使用示例
async def main():
    publisher = AsyncNewsPublisher()
    
    # 添加观察者
    publisher.attach(AsyncEmailNotifier("user@example.com"))
    publisher.attach(AsyncSMSNotifier("13800138000"))
    
    # 发布新闻，触发通知
    await publisher.publish_news("Python 3.11发布，性能提升25%！")
    
    # 移除一个观察者
    print("\n移除短信通知者...")
    publisher.detach(AsyncSMSNotifier("13800138000"))  # 注意：这里实际上是创建了一个新对象，在实际应用中应该保存对象引用
    
    # 再次发布新闻
    await publisher.publish_news("Python 3.12将支持更多新特性！")

# 运行主协程
asyncio.run(main())
```

### 策略模式

策略模式定义了一系列算法，并使它们可以互相替换，让算法的变化独立于使用它的客户端。

```python
from abc import ABC, abstractmethod
from typing import List

# 抽象策略
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

# 具体策略
class BubbleSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("使用冒泡排序")
        result = data.copy()
        n = len(result)
        for i in range(n):
            for j in range(0, n - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result

class QuickSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("使用快速排序")
        result = data.copy()
        if len(result) <= 1:
            return result
        
        pivot = result[len(result) // 2]
        left = [x for x in result if x < pivot]
        middle = [x for x in result if x == pivot]
        right = [x for x in result if x > pivot]
        
        return self.sort(left) + middle + self.sort(right)

class MergeSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("使用归并排序")
        result = data.copy()
        if len(result) <= 1:
            return result
        
        mid = len(result) // 2
        left = self.sort(result[:mid])
        right = self.sort(result[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left: List[int], right: List[int]) -> List[int]:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

# 上下文
class Sorter:
    def __init__(self, strategy: SortStrategy = None):
        self._strategy = strategy or BubbleSortStrategy()
    
    @property
    def strategy(self) -> SortStrategy:
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy
    
    def sort(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)

# 使用示例
data = [5, 3, 8, 4, 6]

# 使用默认策略（冒泡排序）
sorter = Sorter()
print(sorter.sort(data))  # 输出：使用冒泡排序\n[3, 4, 5, 6, 8]

# 切换到快速排序
sorter.strategy = QuickSortStrategy()
print(sorter.sort(data))  # 输出：使用快速排序\n[3, 4, 5, 6, 8]

# 切换到归并排序
sorter.strategy = MergeSortStrategy()
print(sorter.sort(data))  # 输出：使用归并排序\n[3, 4, 5, 6, 8]
```

### 异步策略模式

```python
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# 异步策略接口
class AsyncPaymentStrategy(ABC):
    @abstractmethod
    async def pay(self, amount: float) -> Dict[str, Any]:
        pass

# 具体异步策略
class AsyncCreditCardPayment(AsyncPaymentStrategy):
    def __init__(self, card_number: str, expiry: str, cvv: str):
        self.card_number = card_number
        self.expiry = expiry
        self.cvv = cvv
    
    async def pay(self, amount: float) -> Dict[str, Any]:
        print(f"使用信用卡支付 {amount} 元...")
        # 模拟信用卡支付处理
        await asyncio.sleep(2)
        last_four = self.card_number[-4:]
        return {
            "status": "success",
            "amount": amount,
            "method": "credit_card",
            "card": f"****-****-****-{last_four}",
            "transaction_id": "cc_123456"
        }

class AsyncAlipayPayment(AsyncPaymentStrategy):
    def __init__(self, alipay_id: str):
        self.alipay_id = alipay_id
    
    async def pay(self, amount: float) -> Dict[str, Any]:
        print(f"使用支付宝支付 {amount} 元...")
        # 模拟支付宝支付处理
        await asyncio.sleep(1.5)
        return {
            "status": "success",
            "amount": amount,
            "method": "alipay",
            "account": self.alipay_id,
            "transaction_id": "ap_789012"
        }

class AsyncWeChatPayment(AsyncPaymentStrategy):
    def __init__(self, wechat_id: str):
        self.wechat_id = wechat_id
    
    async def pay(self, amount: float) -> Dict[str, Any]:
        print(f"使用微信支付 {amount} 元...")
        # 模拟微信支付处理
        await asyncio.sleep(1)
        return {
            "status": "success",
            "amount": amount,
            "method": "wechat",
            "account": self.wechat_id,
            "transaction_id": "wx_345678"
        }

# 上下文
class AsyncPaymentProcessor:
    def __init__(self, strategy: AsyncPaymentStrategy = None):
        self.strategy = strategy
    
    async def process_payment(self, amount: float) -> Dict[str, Any]:
        if not self.strategy:
            raise ValueError("未设置支付策略")
        
        result = await self.strategy.pay(amount)
        print(f"支付完成: {result}")
        return result

# 使用示例
async def main():
    # 创建支付处理器
    processor = AsyncPaymentProcessor()
    
    # 使用信用卡支付
    processor.strategy = AsyncCreditCardPayment("1234-5678-9012-3456", "12/25", "123")
    await processor.process_payment(100.50)
    
    print("\n切换支付方式...")
    
    # 使用支付宝支付
    processor.strategy = AsyncAlipayPayment("user@example.com")
    await processor.process_payment(200.75)
    
    print("\n切换支付方式...")
    
    # 使用微信支付
    processor.strategy = AsyncWeChatPayment("wxid_12345")
    await processor.process_payment(150.25)

# 运行主协程
asyncio.run(main())
```

### 命令模式

命令模式将请求封装为一个对象，从而使你可以用不同的请求对客户进行参数化，对请求排队或记录请求日志，以及支持可撤销的操作。

```python
from abc import ABC, abstractmethod
from typing import List, Optional

# 命令接口
class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass
    
    @abstractmethod
    def undo(self) -> None:
        pass

# 接收者
class TextEditor:
    def __init__(self):
        self.text = ""
    
    def insert_text(self, text: str) -> None:
        self.text += text
        print(f"插入文本: '{text}'")
        print(f"当前文本: '{self.text}'")
    
    def delete_text(self, length: int) -> str:
        if length <= 0 or not self.text:
            return ""
        
        deleted = self.text[-length:]
        self.text = self.text[:-length]
        print(f"删除文本: '{deleted}'")
        print(f"当前文本: '{self.text}'")
        return deleted

# 具体命令
class InsertTextCommand(Command):
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text
    
    def execute(self) -> None:
        self.editor.insert_text(self.text)
    
    def undo(self) -> None:
        self.editor.delete_text(len(self.text))

class DeleteTextCommand(Command):
    def __init__(self, editor: TextEditor, length: int):
        self.editor = editor
        self.length = length
        self.deleted_text: Optional[str] = None
    
    def execute(self) -> None:
        self.deleted_text = self.editor.delete_text(self.length)
    
    def undo(self) -> None:
        if self.deleted_text:
            self.editor.insert_text(self.deleted_text)

# 调用者
class CommandHistory:
    def __init__(self):
        self.history: List[Command] = []
    
    def execute(self, command: Command) -> None:
        command.execute()
        self.history.append(command)
    
    def undo_last(self) -> None:
        if self.history:
            command = self.history.pop()
            command.undo()

# 使用示例
editor = TextEditor()
history = CommandHistory()

# 执行命令
history.execute(InsertTextCommand(editor, "Hello, "))
history.execute(InsertTextCommand(editor, "World!"))
history.execute(DeleteTextCommand(editor, 1))  # 删除感叹号
history.execute(InsertTextCommand(editor, "Python!"))

# 撤销操作
print("\n开始撤销操作:")
history.undo_last()  # 撤销添加"Python!"
history.undo_last()  # 撤销删除感叹号
history.undo_last()  # 撤销添加"World!"
```

## 备忘录模式

备忘录模式在不破坏封装性的前提下，捕获一个对象的内部状态，并在该对象之外保存这个状态，以便在需要时能将该对象恢复到原先保存的状态。

```python
from typing import List, Any
import copy

# 备忘录
class Memento:
    def __init__(self, state: Any):
        self._state = copy.deepcopy(state)
    
    def get_state(self) -> Any:
        return copy.deepcopy(self._state)

# 发起人
class Editor:
    def __init__(self):
        self._content = ""
    
    def type(self, words: str) -> None:
        self._content += words
        print(f"当前文本: {self._content}")
    
    def get_content(self) -> str:
        return self._content
    
    def save(self) -> Memento:
        return Memento(self._content)
    
    def restore(self, memento: Memento) -> None:
        self._content = memento.get_state()
        print(f"恢复到: {self._content}")

# 管理者
class History:
    def __init__(self):
        self._mementos: List[Memento] = []
    
    def push(self, memento: Memento) -> None:
        self._mementos.append(memento)
    
    def pop(self) -> Memento:
        return self._mementos.pop()

# 使用示例
editor = Editor()
history = History()

# 编辑文本并保存状态
editor.type("Hello, ")
history.push(editor.save())

editor.type("World!")
history.push(editor.save())

editor.type(" Welcome to Python.")

# 撤销操作
print("\n执行撤销:")
editor.restore(history.pop())

print("\n再次撤销:")
editor.restore(history.pop())
```

## 组合模式

组合模式将对象组合成树形结构以表示"部分-整体"的层次结构，使得用户对单个对象和组合对象的使用具有一致性。

```python
from abc import ABC, abstractmethod
from typing import List

# 组件接口
class FileSystemComponent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def display(self, indent: str = "") -> None:
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        pass

# 叶子节点
class File(FileSystemComponent):
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size = size
    
    def display(self, indent: str = "") -> None:
        print(f"{indent}文件: {self.name} ({self.size} KB)")
    
    def get_size(self) -> int:
        return self.size

# 组合节点
class Directory(FileSystemComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.children: List[FileSystemComponent] = []
    
    def add(self, component: FileSystemComponent) -> None:
        self.children.append(component)
    
    def remove(self, component: FileSystemComponent) -> None:
        self.children.remove(component)
    
    def display(self, indent: str = "") -> None:
        print(f"{indent}目录: {self.name} ({self.get_size()} KB)")
        for child in self.children:
            child.display(indent + "  ")
    
    def get_size(self) -> int:
        return sum(child.get_size() for child in self.children)

# 使用示例
root = Directory("root")

docs = Directory("docs")
docs.add(File("document.txt", 10))
docs.add(File("manual.pdf", 50))

src = Directory("src")
src.add(File("main.py", 5))
src.add(File("utils.py", 8))

lib = Directory("lib")
lib.add(File("library.jar", 100))

src.add(lib)  # 嵌套目录
root.add(docs)
root.add(src)
root.add(File("README.md", 2))

# 显示文件系统结构
root.display()
```

## 协程与异步编程

### 协程基础

协程是可以在执行过程中被挂起和恢复的函数，使用`async`和`await`关键字定义和使用。

```python
import asyncio

async def hello_world():
    print("Hello")
    await asyncio.sleep(1)  # 模拟IO操作，不阻塞事件循环
    print("World")

async def main():
    await hello_world()

asyncio.run(main())
```

### 异步迭代器

```python
import asyncio
from typing import AsyncIterator

class AsyncDataStream:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.current = start
    
    def __aiter__(self) -> 'AsyncDataStream':
        return self
    
    async def __anext__(self) -> int:
        if self.current >= self.end:
            raise StopAsyncIteration
        
        await asyncio.sleep(0.5)  # 模拟异步操作
        value = self.current
        self.current += 1
        return value

async def main():
    async for value in AsyncDataStream(1, 5):
        print(f"获取值: {value}")

# 运行主协程
asyncio.run(main())
```

### 异步生成器

使用异步生成器简化异步迭代器的实现。

```python
import asyncio
from typing import AsyncGenerator

async def async_range(start: int, end: int) -> AsyncGenerator[int, None]:
    for i in range(start, end):
        await asyncio.sleep(0.5)  # 模拟异步操作
        yield i

async def main():
    async for value in async_range(1, 5):
        print(f"获取值: {value}")

# 运行主协程
asyncio.run(main())
```

### 异步队列

使用`asyncio.Queue`实现生产者-消费者模式。

```python
import asyncio
import random
from typing import List

async def producer(queue: asyncio.Queue, id: int) -> None:
    for i in range(5):
        item = f"Producer {id} - Item {i}"
        await queue.put(item)
        print(f"生产: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))

async def consumer(queue: asyncio.Queue, id: int) -> None:
    while True:
        item = await queue.get()
        print(f"消费者 {id} 消费: {item}")
        await asyncio.sleep(random.uniform(0.2, 0.6))
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)
    
    # 创建生产者和消费者
    producers = [asyncio.create_task(producer(queue, i)) for i in range(3)]
    consumers = [asyncio.create_task(consumer(queue, i)) for i in range(2)]
    
    # 等待所有生产者完成
    await asyncio.gather(*producers)
    
    # 等待队列清空
    await queue.join()
    
    # 取消消费者任务
    for c in consumers:
        c.cancel()

# 运行主协程
asyncio.run(main())
```

## 高级设计模式

### 状态模式

状态模式允许对象在内部状态改变时改变它的行为，对象看起来好像修改了它的类。

```python
from abc import ABC, abstractmethod

# 状态接口
class State(ABC):
    @abstractmethod
    def handle(self) -> None:
        pass
    
    @abstractmethod
    def next_state(self, context: 'Context') -> None:
        pass

# 具体状态
class ConcreteStateA(State):
    def handle(self) -> None:
        print("状态A的处理逻辑")
    
    def next_state(self, context: 'Context') -> None:
        print("从状态A切换到状态B")
        context.state = ConcreteStateB()

class ConcreteStateB(State):
    def handle(self) -> None:
        print("状态B的处理逻辑")
    
    def next_state(self, context: 'Context') -> None:
        print("从状态B切换到状态C")
        context.state = ConcreteStateC()

class ConcreteStateC(State):
    def handle(self) -> None:
        print("状态C的处理逻辑")
    
    def next_state(self, context: 'Context') -> None:
        print("从状态C切换到状态A")
        context.state = ConcreteStateA()

# 上下文
class Context:
    def __init__(self):
        self.state: State = ConcreteStateA()
    
    def request(self) -> None:
        self.state.handle()
        self.state.next_state(self)

# 使用示例
context = Context()

# 第一次请求，使用状态A
context.request()

# 第二次请求，使用状态B
context.request()

# 第三次请求，使用状态C
context.request()

# 第四次请求，回到状态A
context.request()
```

### 访问者模式

访问者模式表示一个作用于某对象结构中的各元素的操作，它使你可以在不改变各元素的类的前提下定义作用于这些元素的新操作。

```python
from abc import ABC, abstractmethod
from typing import List

# 访问者接口
class Visitor(ABC):
    @abstractmethod
    def visit_circle(self, circle: 'Circle') -> None:
        pass
    
    @abstractmethod
    def visit_rectangle(self, rectangle: 'Rectangle') -> None:
        pass

# 元素接口
class Shape(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass

# 具体元素
class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_circle(self)

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def accept(self, visitor: Visitor) -> None:
        visitor.visit_rectangle(self)

# 具体访问者
class AreaCalculator(Visitor):
    def __init__(self):
        self.total_area = 0
    
    def visit_circle(self, circle: Circle) -> None:
        area = 3.14 * circle.radius ** 2
        print(f"圆的面积: {area:.2f}")
        self.total_area += area
    
    def visit_rectangle(self, rectangle: Rectangle) -> None:
        area = rectangle.width * rectangle.height
        print(f"矩形的面积: {area:.2f}")
        self.total_area += area

class PerimeterCalculator(Visitor):
    def __init__(self):
        self.total_perimeter = 0
    
    def visit_circle(self, circle: Circle) -> None:
        perimeter = 2 * 3.14 * circle.radius
        print(f"圆的周长: {perimeter:.2f}")
        self.total_perimeter += perimeter
    
    def visit_rectangle(self, rectangle: Rectangle) -> None:
        perimeter = 2 * (rectangle.width + rectangle.height)
        print(f"矩形的周长: {perimeter:.2f}")
        self.total_perimeter += perimeter

# 使用示例
shapes: List[Shape] = [
    Circle(5),
    Rectangle(4, 6),
    Circle(3)
]

# 计算面积
area_calculator = AreaCalculator()
for shape in shapes:
    shape.accept(area_calculator)
print(f"总面积: {area_calculator.total_area:.2f}")

# 计算周长
perimeter_calculator = PerimeterCalculator()
for shape in shapes:
    shape.accept(perimeter_calculator)
print(f"总周长: {perimeter_calculator.total_perimeter:.2f}")
```

## 高级异步模式

### 异步工厂模式

```python
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Type

# 抽象产品
class AsyncProduct(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

# 具体产品
class AsyncDatabaseConnector(AsyncProduct):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False
    
    async def initialize(self) -> None:
        print(f"连接到数据库: {self.connection_string}")
        await asyncio.sleep(1)  # 模拟连接过程
        self.is_connected = True
        print("数据库连接成功")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("数据库未连接")
        
        print(f"处理数据: {data}")
        await asyncio.sleep(0.5)  # 模拟数据处理
        return {"result": "数据已保存", "id": 12345}

class AsyncAPIClient(AsyncProduct):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.is_initialized = False
    
    async def initialize(self) -> None:
        print(f"初始化API客户端: {self.endpoint}")
        await asyncio.sleep(0.8)  # 模拟初始化过程
        self.is_initialized = True
        print("API客户端初始化成功")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("API客户端未初始化")
        
        print(f"发送API请求: {data}")
        await asyncio.sleep(1.2)  # 模拟API请求
        return {"status": 200, "response": {"message": "请求成功"}}

# 异步工厂
class AsyncProductFactory:
    @staticmethod
    async def create_product(product_type: str, **kwargs) -> AsyncProduct:
        if product_type == "database":
            product = AsyncDatabaseConnector(kwargs.get("connection_string", "default_connection"))
        elif product_type == "api":
            product = AsyncAPIClient(kwargs.get("api_key", ""), kwargs.get("endpoint", ""))
        else:
            raise ValueError(f"未知的产品类型: {product_type}")
        
        # 初始化产品
        await product.initialize()
        return product

# 使用示例
async def main():
    # 创建数据库连接器
    db = await AsyncProductFactory.create_product(
        "database", 
        connection_string="postgresql://user:pass@localhost/db"
    )
    db_result = await db.process({"table": "users", "data": {"name": "张三", "age": 30}})
    print(f"数据库处理结果: {db_result}\n")
    
    # 创建API客户端
    api = await AsyncProductFactory.create_product(
        "api", 
        api_key="api_key_12345", 
        endpoint="https://api.example.com/v1"
    )
    api_result = await api.process({"method": "GET", "path": "/users"})
    print(f"API处理结果: {api_result}")

# 运行主协程
asyncio.run(main())
```

## 总结

本文档整合了Python高级设计模式与异步编程的核心内容，涵盖了：

1. **经典设计模式的Python实现**：观察者模式、策略模式、命令模式、备忘录模式、组合模式、状态模式和访问者模式等。

2. **异步编程技术**：协程基础、异步迭代器、异步生成器、异步队列等。

3. **异步设计模式**：异步观察者模式、异步策略模式、异步工厂模式等。

通过学习和应用这些模式，可以编写出更加灵活、可维护和高效的Python代码。设计模式提供了解决特定问题的通用方案，而异步编程则提供了处理IO密集型任务的高效方式。将两者结合，可以构建出既结构良好又性能优异的应用程序。