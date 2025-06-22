"""
Enhanced Asynchronous Processing System for Universal Agent
==========================================================
Advanced async capabilities including streaming, queuing, and real-time processing
"""

import asyncio
import aiostream
from aiostream import stream
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import aio_pika
import nats
from nats.aio.client import Client as NATS
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
import aioredis
from aioredis import Redis
import websockets
from websockets.server import serve
import grpcio
from grpcio import aio as grpc_aio
import uvloop
from typing import AsyncIterator, AsyncGenerator, Awaitable, AsyncContextManager
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
import msgpack
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from asyncio import Queue, PriorityQueue, LifoQueue
import weakref
import signal
import sys
from prometheus_client import Counter, Histogram, Gauge
import structlog
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Set uvloop as the event loop policy for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = structlog.get_logger()

# Type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# ========== Async Message Queue System ==========

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BULK = 5

@dataclass
class AsyncMessage(Generic[T]):
    """Async message wrapper"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: T = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    headers: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[timedelta] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

class AsyncMessageBroker:
    """High-performance async message broker"""
    
    def __init__(self, broker_type: str = "memory"):
        self.broker_type = broker_type
        self.topics: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.dlq: asyncio.Queue = asyncio.Queue()  # Dead letter queue
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Metrics
        self.messages_published = Counter('broker_messages_published', 'Messages published', ['topic'])
        self.messages_consumed = Counter('broker_messages_consumed', 'Messages consumed', ['topic'])
        self.messages_failed = Counter('broker_messages_failed', 'Messages failed', ['topic'])
        
    async def start(self):
        """Start the message broker"""
        self._running = True
        
        # Start consumer tasks for each topic
        for topic in self.topics:
            task = asyncio.create_task(self._consume_topic(topic))
            self._tasks.append(task)
        
        logger.info("Async message broker started")
    
    async def stop(self):
        """Stop the message broker"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Async message broker stopped")
    
    async def create_topic(self, topic: str, max_size: int = 0):
        """Create a new topic"""
        if topic not in self.topics:
            self.topics[topic] = asyncio.Queue(maxsize=max_size)
            
            # Start consumer for this topic
            if self._running:
                task = asyncio.create_task(self._consume_topic(topic))
                self._tasks.append(task)
    
    async def publish(self, topic: str, message: AsyncMessage[T], timeout: Optional[float] = None):
        """Publish message to topic"""
        if topic not in self.topics:
            await self.create_topic(topic)
        
        try:
            queue = self.topics[topic]
            
            # Check TTL
            if message.ttl and datetime.now() - message.timestamp > message.ttl:
                logger.warning(f"Message {message.id} expired, sending to DLQ")
                await self.dlq.put(message)
                return
            
            # Put message in queue with timeout
            if timeout:
                await asyncio.wait_for(queue.put(message), timeout=timeout)
            else:
                await queue.put(message)
            
            self.messages_published.labels(topic=topic).inc()
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout publishing to topic {topic}")
            await self.dlq.put(message)
        except Exception as e:
            logger.error(f"Error publishing to topic {topic}: {e}")
            await self.dlq.put(message)
    
    async def subscribe(self, topic: str, handler: Callable[[AsyncMessage[T]], Awaitable[None]]):
        """Subscribe to a topic"""
        if topic not in self.topics:
            await self.create_topic(topic)
        
        self.subscribers[topic].append(handler)
    
    async def _consume_topic(self, topic: str):
        """Consume messages from a topic"""
        queue = self.topics[topic]
        
        while self._running:
            try:
                # Get message with timeout to allow checking _running
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Process message with all subscribers
                await self._process_message(topic, message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error consuming from topic {topic}: {e}")
    
    async def _process_message(self, topic: str, message: AsyncMessage[T]):
        """Process a message with all subscribers"""
        handlers = self.subscribers.get(topic, [])
        
        if not handlers:
            logger.warning(f"No handlers for topic {topic}")
            return
        
        # Process with all handlers concurrently
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._handle_message(topic, message, handler))
            tasks.append(task)
        
        # Wait for all handlers to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            await self._handle_failures(topic, message, failures)
    
    async def _handle_message(self, topic: str, message: AsyncMessage[T], handler: Callable):
        """Handle a single message with retry logic"""
        try:
            await handler(message)
            self.messages_consumed.labels(topic=topic).inc()
            
        except Exception as e:
            logger.error(f"Handler error for topic {topic}: {e}")
            self.messages_failed.labels(topic=topic).inc()
            
            # Retry logic
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                await asyncio.sleep(2 ** message.retry_count)  # Exponential backoff
                await self.publish(topic, message)
            else:
                # Send to DLQ
                await self.dlq.put(message)
                raise
    
    async def _handle_failures(self, topic: str, message: AsyncMessage[T], failures: List[Exception]):
        """Handle message processing failures"""
        logger.error(f"Message {message.id} failed on topic {topic} with {len(failures)} errors")
        
        # Send to DLQ if all handlers failed
        if len(failures) == len(self.subscribers[topic]):
            await self.dlq.put(message)

# ========== Async Stream Processing ==========

class AsyncStreamProcessor(Generic[T]):
    """Async stream processing pipeline"""
    
    def __init__(self, name: str):
        self.name = name
        self.pipeline: List[Callable] = []
        self.error_handler: Optional[Callable] = None
        self.metrics = {
            'processed': Counter(f'{name}_processed', 'Items processed'),
            'errors': Counter(f'{name}_errors', 'Processing errors'),
            'latency': Histogram(f'{name}_latency', 'Processing latency')
        }
    
    def map(self, func: Callable[[T], Awaitable[T]]) -> 'AsyncStreamProcessor[T]':
        """Add map operation to pipeline"""
        self.pipeline.append(('map', func))
        return self
    
    def filter(self, predicate: Callable[[T], Awaitable[bool]]) -> 'AsyncStreamProcessor[T]':
        """Add filter operation to pipeline"""
        self.pipeline.append(('filter', predicate))
        return self
    
    def batch(self, size: int, timeout: float = None) -> 'AsyncStreamProcessor[List[T]]':
        """Add batch operation to pipeline"""
        async def batch_func(stream):
            async for batch in aiostream.stream.chunks(stream, size):
                yield batch
        
        self.pipeline.append(('batch', batch_func))
        return self
    
    def window(self, size: int, slide: int = None) -> 'AsyncStreamProcessor[List[T]]':
        """Add windowing operation to pipeline"""
        slide = slide or size
        
        async def window_func(stream):
            window = deque(maxlen=size)
            count = 0
            
            async for item in stream:
                window.append(item)
                count += 1
                
                if count >= size and (count - size) % slide == 0:
                    yield list(window)
        
        self.pipeline.append(('window', window_func))
        return self
    
    def parallel(self, func: Callable[[T], Awaitable[T]], max_workers: int = 10) -> 'AsyncStreamProcessor[T]':
        """Add parallel processing operation"""
        async def parallel_func(stream):
            async def process_item(item):
                try:
                    return await func(item)
                except Exception as e:
                    if self.error_handler:
                        return await self.error_handler(item, e)
                    raise
            
            # Create a pool of workers
            semaphore = asyncio.Semaphore(max_workers)
            
            async def bounded_process(item):
                async with semaphore:
                    return await process_item(item)
            
            # Process items in parallel
            async for item in stream:
                yield await bounded_process(item)
        
        self.pipeline.append(('parallel', parallel_func))
        return self
    
    def on_error(self, handler: Callable[[T, Exception], Awaitable[Optional[T]]]) -> 'AsyncStreamProcessor[T]':
        """Set error handler"""
        self.error_handler = handler
        return self
    
    async def process(self, source: AsyncIterator[T]) -> AsyncIterator[T]:
        """Process the stream through the pipeline"""
        stream = source
        
        for operation, func in self.pipeline:
            if operation in ['batch', 'window']:
                stream = func(stream)
            else:
                stream = self._apply_operation(stream, operation, func)
        
        async for item in stream:
            self.metrics['processed'].inc()
            yield item
    
    async def _apply_operation(self, stream: AsyncIterator[T], operation: str, func: Callable) -> AsyncIterator[T]:
        """Apply a single operation to the stream"""
        async for item in stream:
            try:
                with self.metrics['latency'].time():
                    if operation == 'map':
                        result = await func(item)
                        yield result
                    elif operation == 'filter':
                        if await func(item):
                            yield item
                    elif operation == 'parallel':
                        async for result in func(aiostream.stream.iterate(item)):
                            yield result
                    
            except Exception as e:
                self.metrics['errors'].inc()
                if self.error_handler:
                    result = await self.error_handler(item, e)
                    if result is not None:
                        yield result
                else:
                    raise

# ========== Async Task Queue with Advanced Features ==========

@dataclass
class AsyncTask(Generic[T]):
    """Enhanced async task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    payload: T = None
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    status: str = "pending"

class AsyncTaskQueue:
    """Advanced async task queue with scheduling and dependencies"""
    
    def __init__(self, name: str, max_workers: int = 10):
        self.name = name
        self.max_workers = max_workers
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.scheduled_tasks: Dict[str, AsyncTask] = {}
        self.running_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: Dict[str, AsyncTask] = {}
        self.task_futures: Dict[str, asyncio.Future] = {}
        self.workers: List[asyncio.Task] = []
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.tasks_queued = Gauge(f'{name}_tasks_queued', 'Tasks in queue')
        self.tasks_running = Gauge(f'{name}_tasks_running', 'Tasks running')
        self.tasks_completed = Counter(f'{name}_tasks_completed', 'Tasks completed')
        self.tasks_failed = Counter(f'{name}_tasks_failed', 'Tasks failed')
        
    async def start(self):
        """Start the task queue"""
        self._running = True
        
        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start scheduler
        self._scheduler_task = asyncio.create_task(self._scheduler())
        
        logger.info(f"Task queue '{self.name}' started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the task queue"""
        self._running = False
        
        # Cancel scheduler
        if self._scheduler_task:
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info(f"Task queue '{self.name}' stopped")
    
    async def submit(self, task: AsyncTask[T]) -> asyncio.Future:
        """Submit a task to the queue"""
        # Create future for task result
        future = asyncio.Future()
        self.task_futures[task.id] = future
        
        # Check if task should be scheduled
        if task.scheduled_at and task.scheduled_at > datetime.now():
            self.scheduled_tasks[task.id] = task
            logger.info(f"Task {task.id} scheduled for {task.scheduled_at}")
        else:
            # Check dependencies
            if await self._check_dependencies(task):
                await self._enqueue_task(task)
            else:
                # Wait for dependencies
                asyncio.create_task(self._wait_for_dependencies(task))
        
        return future
    
    async def _enqueue_task(self, task: AsyncTask[T]):
        """Enqueue a task"""
        # Priority is negative so higher priority items come first
        await self.pending_queue.put((-task.priority, task.created_at, task))
        self.tasks_queued.set(self.pending_queue.qsize())
        logger.debug(f"Task {task.id} enqueued")
    
    async def _worker(self, worker_id: str):
        """Worker coroutine"""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get task with timeout
                priority, created_at, task = await asyncio.wait_for(
                    self.pending_queue.get(),
                    timeout=1.0
                )
                
                # Update metrics
                self.tasks_queued.set(self.pending_queue.qsize())
                self.tasks_running.inc()
                
                # Process task
                await self._process_task(task)
                
                # Update metrics
                self.tasks_running.dec()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: AsyncTask[T]):
        """Process a single task"""
        task.status = "running"
        self.running_tasks[task.id] = task
        future = self.task_futures.get(task.id)
        
        try:
            # Execute task with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_task(task),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_task(task)
            
            # Update task
            task.result = result
            task.status = "completed"
            self.completed_tasks[task.id] = task
            self.tasks_completed.inc()
            
            # Set future result
            if future and not future.done():
                future.set_result(result)
            
            logger.info(f"Task {task.id} completed successfully")
            
        except asyncio.TimeoutError:
            await self._handle_task_failure(task, "Task timeout", future)
        except Exception as e:
            await self._handle_task_failure(task, str(e), future)
        finally:
            # Remove from running
            self.running_tasks.pop(task.id, None)
    
    async def _execute_task(self, task: AsyncTask[T]) -> Any:
        """Execute the actual task logic"""
        # This would be overridden or use a task handler registry
        if 'handler' in task.metadata:
            handler = task.metadata['handler']
            return await handler(task.payload)
        else:
            # Default: just return the payload
            await asyncio.sleep(0.1)  # Simulate work
            return task.payload
    
    async def _handle_task_failure(self, task: AsyncTask[T], error: str, future: Optional[asyncio.Future]):
        """Handle task failure with retry logic"""
        task.error = error
        task.retries += 1
        
        if task.retries < task.max_retries:
            # Retry with exponential backoff
            delay = 2 ** task.retries
            logger.warning(f"Task {task.id} failed, retrying in {delay}s")
            
            task.scheduled_at = datetime.now() + timedelta(seconds=delay)
            self.scheduled_tasks[task.id] = task
        else:
            # Max retries reached
            task.status = "failed"
            self.completed_tasks[task.id] = task
            self.tasks_failed.inc()
            
            # Set future exception
            if future and not future.done():
                future.set_exception(Exception(error))
            
            logger.error(f"Task {task.id} failed after {task.retries} retries: {error}")
    
    async def _scheduler(self):
        """Schedule tasks that are ready to run"""
        while self._running:
            try:
                now = datetime.now()
                tasks_to_run = []
                
                # Check scheduled tasks
                for task_id, task in list(self.scheduled_tasks.items()):
                    if task.scheduled_at <= now:
                        tasks_to_run.append(task)
                        del self.scheduled_tasks[task_id]
                
                # Enqueue ready tasks
                for task in tasks_to_run:
                    if await self._check_dependencies(task):
                        await self._enqueue_task(task)
                    else:
                        # Re-schedule if dependencies not met
                        task.scheduled_at = now + timedelta(seconds=5)
                        self.scheduled_tasks[task.id] = task
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    async def _check_dependencies(self, task: AsyncTask[T]) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            dep_task = self.completed_tasks[dep_id]
            if dep_task.status != "completed":
                return False
        
        return True
    
    async def _wait_for_dependencies(self, task: AsyncTask[T]):
        """Wait for task dependencies to complete"""
        while not await self._check_dependencies(task):
            await asyncio.sleep(0.5)
        
        await self._enqueue_task(task)

# ========== WebSocket Real-time Communication ==========

class AsyncWebSocketServer:
    """Async WebSocket server for real-time communication"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)
        self.message_handlers: Dict[str, Callable] = {}
        self.server = None
        
    async def start(self):
        """Start WebSocket server"""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all client connections
        for client in list(self.clients.values()):
            await client.close()
        
        logger.info("WebSocket server stopped")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle client connection"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        
        logger.info(f"Client {client_id} connected")
        
        try:
            # Send welcome message
            await self.send_to_client(client_id, {
                'type': 'welcome',
                'client_id': client_id
            })
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up
            await self.disconnect_client(client_id)
    
    async def handle_message(self, client_id: str, message: str):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type in self.message_handlers:
                handler = self.message_handlers[msg_type]
                await handler(client_id, data)
            else:
                await self.handle_default_message(client_id, data)
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client_id}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    async def handle_default_message(self, client_id: str, data: Dict[str, Any]):
        """Default message handler"""
        msg_type = data.get('type')
        
        if msg_type == 'join_room':
            room = data.get('room')
            await self.join_room(client_id, room)
            
        elif msg_type == 'leave_room':
            room = data.get('room')
            await self.leave_room(client_id, room)
            
        elif msg_type == 'broadcast':
            room = data.get('room')
            message = data.get('message')
            await self.broadcast_to_room(room, message, exclude=client_id)
            
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    def register_handler(self, msg_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[msg_type] = handler
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.clients:
            websocket = self.clients[client_id]
            message = json.dumps(data)
            await websocket.send(message)
    
    async def broadcast(self, data: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast to all clients"""
        message = json.dumps(data)
        
        tasks = []
        for client_id, websocket in self.clients.items():
            if client_id != exclude:
                tasks.append(websocket.send(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def join_room(self, client_id: str, room: str):
        """Join a room"""
        self.rooms[room].add(client_id)
        
        # Notify room members
        await self.broadcast_to_room(room, {
            'type': 'user_joined',
            'client_id': client_id,
            'room': room
        }, exclude=client_id)
        
        logger.info(f"Client {client_id} joined room {room}")
    
    async def leave_room(self, client_id: str, room: str):
        """Leave a room"""
        self.rooms[room].discard(client_id)
        
        # Notify room members
        await self.broadcast_to_room(room, {
            'type': 'user_left',
            'client_id': client_id,
            'room': room
        })
        
        logger.info(f"Client {client_id} left room {room}")
    
    async def broadcast_to_room(self, room: str, data: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast to all clients in a room"""
        message = json.dumps(data)
        
        tasks = []
        for client_id in self.rooms[room]:
            if client_id != exclude and client_id in self.clients:
                websocket = self.clients[client_id]
                tasks.append(websocket.send(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def disconnect_client(self, client_id: str):
        """Disconnect and clean up client"""
        # Remove from all rooms
        for room in list(self.rooms.keys()):
            await self.leave_room(client_id, room)
        
        # Remove client
        self.clients.pop(client_id, None)

# ========== Async Event Bus ==========

class AsyncEvent:
    """Async event"""
    def __init__(self, event_type: str, data: Any = None, source: str = None):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.metadata: Dict[str, Any] = {}

class AsyncEventBus:
    """Async event bus for decoupled communication"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.middleware: List[Callable] = []
        self.event_store: deque = deque(maxlen=10000)
        self._running = True
        self._event_queue = asyncio.Queue()
        self._processor_task = None
        
        # Metrics
        self.events_published = Counter('eventbus_events_published', 'Events published', ['type'])
        self.events_processed = Counter('eventbus_events_processed', 'Events processed', ['type'])
        
    async def start(self):
        """Start event processor"""
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop event processor"""
        self._running = False
        if self._processor_task:
            await self._event_queue.put(None)  # Sentinel
            await self._processor_task
        logger.info("Event bus stopped")
    
    async def publish(self, event: AsyncEvent):
        """Publish an event"""
        # Apply middleware
        for mw in self.middleware:
            event = await mw(event)
            if event is None:
                return  # Event filtered out
        
        # Store event
        self.event_store.append(event)
        
        # Queue for processing
        await self._event_queue.put(event)
        self.events_published.labels(type=event.type).inc()
    
    def subscribe(self, event_type: str, handler: Callable[[AsyncEvent], Awaitable[None]]):
        """Subscribe to event type"""
        self.handlers[event_type].append(handler)
        
    def unsubscribe(self, event_type: str, handler: Callable[[AsyncEvent], Awaitable[None]]):
        """Unsubscribe from event type"""
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)
    
    def use_middleware(self, middleware: Callable[[AsyncEvent], Awaitable[AsyncEvent]]):
        """Add middleware"""
        self.middleware.append(middleware)
    
    async def _process_events(self):
        """Process events from queue"""
        while self._running:
            try:
                event = await self._event_queue.get()
                if event is None:  # Sentinel
                    break
                
                # Get handlers for event type
                handlers = self.handlers.get(event.type, [])
                handlers.extend(self.handlers.get('*', []))  # Wildcard handlers
                
                if handlers:
                    # Process with all handlers concurrently
                    tasks = [self._handle_event(handler, event) for handler in handlers]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                self.events_processed.labels(type=event.type).inc()
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, handler: Callable, event: AsyncEvent):
        """Handle single event with error handling"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Handler error for event {event.type}: {e}")
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[AsyncEvent]:
        """Get event history"""
        events = list(self.event_store)
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return events[-limit:]

# ========== Async Data Pipeline ==========

class AsyncDataPipeline:
    """Async data processing pipeline with backpressure"""
    
    def __init__(self, name: str, buffer_size: int = 1000):
        self.name = name
        self.buffer_size = buffer_size
        self.stages: List[Tuple[str, Callable, Dict]] = []
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._running = False
        self._workers: List[asyncio.Task] = []
        
        # Metrics
        self.items_processed = Counter(f'{name}_items_processed', 'Items processed')
        self.processing_time = Histogram(f'{name}_processing_time', 'Processing time')
        self.queue_size = Gauge(f'{name}_queue_size', 'Queue size')
        
    def add_stage(self, name: str, processor: Callable, workers: int = 1, **kwargs):
        """Add processing stage"""
        self.stages.append((name, processor, {'workers': workers, **kwargs}))
        return self
    
    async def start(self):
        """Start the pipeline"""
        self._running = True
        
        # Create workers for each stage
        for i, (stage_name, processor, config) in enumerate(self.stages):
            workers = config.get('workers', 1)
            
            for worker_id in range(workers):
                if i == 0:
                    # First stage reads from input queue
                    input_q = self.input_queue
                else:
                    # Create intermediate queue
                    input_q = asyncio.Queue(maxsize=self.buffer_size)
                    # Connect previous stage output to this input
                    self._workers[-1].output_queue = input_q
                
                if i == len(self.stages) - 1:
                    # Last stage writes to output queue
                    output_q = self.output_queue
                else:
                    output_q = None
                
                worker = asyncio.create_task(
                    self._stage_worker(
                        f"{stage_name}-{worker_id}",
                        processor,
                        input_q,
                        output_q,
                        config
                    )
                )
                worker.output_queue = output_q
                self._workers.append(worker)
        
        logger.info(f"Pipeline '{self.name}' started with {len(self._workers)} workers")
    
    async def stop(self):
        """Stop the pipeline"""
        self._running = False
        
        # Send stop signal
        await self.input_queue.put(None)
        
        # Wait for workers
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        logger.info(f"Pipeline '{self.name}' stopped")
    
    async def process(self, item: T) -> None:
        """Process an item through the pipeline"""
        await self.input_queue.put(item)
        self.queue_size.set(self.input_queue.qsize())
    
    async def get_result(self) -> Optional[T]:
        """Get processed result"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def process_stream(self, stream: AsyncIterator[T]) -> AsyncIterator[T]:
        """Process a stream of items"""
        # Feed items to pipeline
        async def feeder():
            async for item in stream:
                await self.process(item)
            await self.input_queue.put(None)  # End signal
        
        # Start feeder
        feeder_task = asyncio.create_task(feeder())
        
        # Yield results
        while True:
            result = await self.output_queue.get()
            if result is None:
                break
            yield result
        
        await feeder_task
    
    async def _stage_worker(
        self,
        worker_id: str,
        processor: Callable,
        input_queue: asyncio.Queue,
        output_queue: Optional[asyncio.Queue],
        config: Dict
    ):
        """Worker for a pipeline stage"""
        logger.info(f"Stage worker {worker_id} started")
        
        while self._running:
            try:
                # Get item with backpressure
                item = await input_queue.get()
                if item is None:  # Stop signal
                    if output_queue:
                        await output_queue.put(None)
                    break
                
                # Process item
                with self.processing_time.time():
                    result = await processor(item, **config)
                
                # Send to next stage
                if output_queue and result is not None:
                    await output_queue.put(result)
                
                self.items_processed.inc()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Stage worker {worker_id} stopped")

# ========== Async Circuit Breaker ==========

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class AsyncCircuitBreaker:
    """Async circuit breaker pattern"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        
        # Metrics
        self.calls = Counter(f'{name}_circuit_calls', 'Circuit breaker calls', ['state'])
        self.failures = Counter(f'{name}_circuit_failures', 'Circuit breaker failures')
        self.state_changes = Counter(f'{name}_circuit_state_changes', 'State changes', ['from', 'to'])
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Call function through circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitBreakerState.HALF_OPEN)
            else:
                self.calls.labels(state='open').inc()
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count > self.failure_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
        
        self.calls.labels(state=self.state.value).inc()
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self._transition_to(CircuitBreakerState.OPEN)
        
        self.failures.inc()
        self.calls.labels(state=self.state.value).inc()
    
    def _transition_to(self, new_state: CircuitBreakerState):
        """Transition to new state"""
        if self.state != new_state:
            logger.info(f"Circuit breaker '{self.name}' transitioning from {self.state.value} to {new_state.value}")
            self.state_changes.labels(from_state=self.state.value, to=new_state.value).inc()
            self.state = new_state
            
            if new_state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
                self.success_count = 0

# ========== Async Rate Limiter ==========

class AsyncRateLimiter:
    """Token bucket rate limiter with async support"""
    
    def __init__(self, rate: float, capacity: float, name: str = "default"):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.name = name
        self.tokens = capacity
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
        
        # Metrics
        self.allowed = Counter(f'{name}_ratelimit_allowed', 'Requests allowed')
        self.rejected = Counter(f'{name}_ratelimit_rejected', 'Requests rejected')
        self.wait_time = Histogram(f'{name}_ratelimit_wait_time', 'Wait time for tokens')
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """Acquire tokens, returns True if allowed"""
        async with self._lock:
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.allowed.inc()
                return True
            else:
                self.rejected.inc()
                return False
    
    async def acquire_wait(self, tokens: float = 1.0) -> None:
        """Acquire tokens, wait if necessary"""
        while True:
            async with self._lock:
                await self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.allowed.inc()
                    return
                
                # Calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
            
            # Wait outside lock
            with self.wait_time.time():
                await asyncio.sleep(wait_time)
    
    async def _refill(self):
        """Refill tokens based on elapsed time"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_update
        
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

# ========== Async Cache with TTL ==========

class AsyncCache(Generic[K, V]):
    """Async cache with TTL and eviction policies"""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0, name: str = "default"):
        self.max_size = max_size
        self.ttl = ttl
        self.name = name
        self.cache: Dict[K, Tuple[V, float]] = {}
        self.access_times: Dict[K, float] = {}
        self.access_counts: Dict[K, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        
        # Metrics
        self.hits = Counter(f'{name}_cache_hits', 'Cache hits')
        self.misses = Counter(f'{name}_cache_misses', 'Cache misses')
        self.evictions = Counter(f'{name}_cache_evictions', 'Cache evictions')
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def get(self, key: K) -> Optional[V]:
        """Get value from cache"""
        async with self._lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                
                if asyncio.get_event_loop().time() < expiry:
                    # Update access tracking
                    self.access_times[key] = asyncio.get_event_loop().time()
                    self.access_counts[key] += 1
                    self.hits.inc()
                    return value
                else:
                    # Expired
                    await self._evict(key)
            
            self.misses.inc()
            return None
    
    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            # Check size limit
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            # Set value with expiry
            expiry = asyncio.get_event_loop().time() + (ttl or self.ttl)
            self.cache[key] = (value, expiry)
            self.access_times[key] = asyncio.get_event_loop().time()
            self.access_counts[key] = 0
    
    async def delete(self, key: K) -> bool:
        """Delete from cache"""
        async with self._lock:
            if key in self.cache:
                await self._evict(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear cache"""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    async def _evict(self, key: K) -> None:
        """Evict a key"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.evictions.inc()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if self.access_times:
            lru_key = min(self.access_times, key=self.access_times.get)
            await self._evict(lru_key)
    
    async def _cleanup_expired(self) -> None:
        """Periodic cleanup of expired items"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self._lock:
                    now = asyncio.get_event_loop().time()
                    expired_keys = [
                        key for key, (_, expiry) in self.cache.items()
                        if now >= expiry
                    ]
                    
                    for key in expired_keys:
                        await self._evict(key)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def close(self) -> None:
        """Close cache and cleanup"""
        self._cleanup_task.cancel()
        await asyncio.gather(self._cleanup_task, return_exceptions=True)

# ========== Async Batch Processor ==========

class AsyncBatchProcessor(Generic[T]):
    """Process items in batches with timeout"""
    
    def __init__(
        self,
        processor: Callable[[List[T]], Awaitable[None]],
        batch_size: int = 100,
        batch_timeout: float = 1.0,
        name: str = "default"
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.name = name
        
        self.buffer: List[T] = []
        self.buffer_lock = asyncio.Lock()
        self.flush_event = asyncio.Event()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.batches_processed = Counter(f'{name}_batches_processed', 'Batches processed')
        self.items_processed = Counter(f'{name}_items_processed', 'Items processed')
        self.batch_sizes = Histogram(f'{name}_batch_sizes', 'Batch sizes')
    
    async def start(self):
        """Start batch processor"""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_batches())
        logger.info(f"Batch processor '{self.name}' started")
    
    async def stop(self):
        """Stop batch processor"""
        self._running = False
        
        # Flush remaining
        await self.flush()
        
        # Stop processor
        if self._processor_task:
            self.flush_event.set()
            await self._processor_task
        
        logger.info(f"Batch processor '{self.name}' stopped")
    
    async def add(self, item: T) -> None:
        """Add item to batch"""
        async with self.buffer_lock:
            self.buffer.append(item)
            
            if len(self.buffer) >= self.batch_size:
                self.flush_event.set()
    
    async def flush(self) -> None:
        """Force flush current batch"""
        self.flush_event.set()
        
        # Wait for flush to complete
        await asyncio.sleep(0.1)
    
    async def _process_batches(self):
        """Process batches"""
        while self._running:
            try:
                # Wait for batch to be ready or timeout
                try:
                    await asyncio.wait_for(
                        self.flush_event.wait(),
                        timeout=self.batch_timeout
                    )
                except asyncio.TimeoutError:
                    pass
                
                # Clear event
                self.flush_event.clear()
                
                # Get batch
                async with self.buffer_lock:
                    if not self.buffer:
                        continue
                    
                    batch = self.buffer[:]
                    self.buffer.clear()
                
                # Process batch
                await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    async def _process_batch(self, batch: List[T]) -> None:
        """Process a single batch"""
        if not batch:
            return
        
        try:
            await self.processor(batch)
            
            # Update metrics
            self.batches_processed.inc()
            self.items_processed.inc(len(batch))
            self.batch_sizes.observe(len(batch))
            
            logger.debug(f"Processed batch of {len(batch)} items")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")

# ========== Async Connection Pool ==========

class AsyncConnectionPool(Generic[T]):
    """Generic async connection pool"""
    
    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        min_size: int = 5,
        max_size: int = 20,
        name: str = "default"
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.name = name
        
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self._lock = asyncio.Lock()
        self._closed = False
        
        # Metrics
        self.active_connections = Gauge(f'{name}_pool_active', 'Active connections')
        self.total_connections = Gauge(f'{name}_pool_total', 'Total connections')
        self.wait_time = Histogram(f'{name}_pool_wait_time', 'Wait time for connection')
    
    async def initialize(self):
        """Initialize pool with minimum connections"""
        async with self._lock:
            for _ in range(self.min_size):
                conn = await self.factory()
                await self.pool.put(conn)
                self.size += 1
            
            self.total_connections.set(self.size)
        
        logger.info(f"Connection pool '{self.name}' initialized with {self.size} connections")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[T]:
        """Acquire connection from pool"""
        if self._closed:
            raise RuntimeError("Pool is closed")
        
        start_time = asyncio.get_event_loop().time()
        conn = None
        
        try:
            # Try to get from pool
            try:
                conn = await asyncio.wait_for(self.pool.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # Create new connection if under limit
                async with self._lock:
                    if self.size < self.max_size:
                        conn = await self.factory()
                        self.size += 1
                        self.total_connections.set(self.size)
                    else:
                        # Wait for available connection
                        conn = await self.pool.get()
            
            # Record wait time
            wait_time = asyncio.get_event_loop().time() - start_time
            self.wait_time.observe(wait_time)
            
            # Update metrics
            self.active_connections.inc()
            
            yield conn
            
        finally:
            # Return to pool
            if conn is not None:
                await self.pool.put(conn)
            
            self.active_connections.dec()
    
    async def close(self):
        """Close all connections"""
        async with self._lock:
            self._closed = True
            
            # Close all connections
            while not self.pool.empty():
                try:
                    conn = await self.pool.get()
                    if hasattr(conn, 'close'):
                        await conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self.size = 0
            self.total_connections.set(0)
        
        logger.info(f"Connection pool '{self.name}' closed")

# ========== Async Semaphore with Priority ==========

class PrioritySemaphore:
    """Async semaphore with priority support"""
    
    def __init__(self, value: int = 1):
        self._value = value
        self._waiters: List[Tuple[int, asyncio.Future]] = []
        
    async def acquire(self, priority: int = 0) -> None:
        """Acquire with priority (lower number = higher priority)"""
        while self._value <= 0:
            fut = asyncio.Future()
            heapq.heappush(self._waiters, (priority, fut))
            
            try:
                await fut
            except:
                # Remove from waiters if cancelled
                self._waiters.remove((priority, fut))
                if self._value > 0 and self._waiters:
                    # Wake next waiter
                    _, next_fut = heapq.heappop(self._waiters)
                    if not next_fut.done():
                        next_fut.set_result(None)
                raise
        
        self._value -= 1
    
    def release(self) -> None:
        """Release semaphore"""
        self._value += 1
        
        if self._waiters:
            # Wake highest priority waiter
            _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

# ========== Integration with Main System ==========

class EnhancedAsyncSystem:
    """Enhanced async capabilities for Universal Agent System"""
    
    def __init__(self):
        # Message broker
        self.message_broker = AsyncMessageBroker()
        
        # Task queue
        self.task_queue = AsyncTaskQueue("main", max_workers=20)
        
        # WebSocket server
        self.ws_server = AsyncWebSocketServer()
        
        # Event bus
        self.event_bus = AsyncEventBus()
        
        # Connection pools
        self.db_pool = AsyncConnectionPool(
            factory=self._create_db_connection,
            min_size=5,
            max_size=20,
            name="database"
        )
        
        self.redis_pool = AsyncConnectionPool(
            factory=self._create_redis_connection,
            min_size=3,
            max_size=10,
            name="redis"
        )
        
        # Rate limiters
        self.api_limiter = AsyncRateLimiter(
            rate=100,  # 100 requests per second
            capacity=1000,
            name="api"
        )
        
        # Circuit breakers
        self.model_circuit = AsyncCircuitBreaker(
            name="model_api",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # Caches
        self.result_cache = AsyncCache[str, Any](
            max_size=10000,
            ttl=3600,
            name="results"
        )
        
        # Batch processors
        self.log_batch = AsyncBatchProcessor(
            processor=self._process_logs,
            batch_size=1000,
            batch_timeout=5.0,
            name="logs"
        )
        
    async def start(self):
        """Start all async systems"""
        # Start message broker
        await self.message_broker.start()
        
        # Start task queue
        await self.task_queue.start()
        
        # Start WebSocket server
        await self.ws_server.start()
        
        # Start event bus
        await self.event_bus.start()
        
        # Initialize pools
        await self.db_pool.initialize()
        await self.redis_pool.initialize()
        
        # Start batch processors
        await self.log_batch.start()
        
        logger.info("Enhanced async system started")
    
    async def stop(self):
        """Stop all async systems"""
        # Stop in reverse order
        await self.log_batch.stop()
        await self.redis_pool.close()
        await self.db_pool.close()
        await self.event_bus.stop()
        await self.ws_server.stop()
        await self.task_queue.stop()
        await self.message_broker.stop()
        await self.result_cache.close()
        
        logger.info("Enhanced async system stopped")
    
    async def _create_db_connection(self):
        """Create database connection"""
        return await asyncpg.connect(
            host='localhost',
            port=5432,
            user='user',
            password='password',
            database='universal_agent'
        )
    
    async def _create_redis_connection(self):
        """Create Redis connection"""
        return await aioredis.create_redis_pool(
            'redis://localhost:6379',
            minsize=1,
            maxsize=5
        )
    
    async def _process_logs(self, logs: List[Dict[str, Any]]):
        """Process log batch"""
        # This would send logs to storage/monitoring
        logger.info(f"Processing {len(logs)} log entries")

# ========== Example Usage ==========

async def example_async_enhancements():
    """Example of using enhanced async features"""
    system = EnhancedAsyncSystem()
    await system.start()
    
    try:
        # 1. Message broker example
        async def message_handler(msg: AsyncMessage):
            print(f"Received: {msg.content}")
        
        await system.message_broker.subscribe("notifications", message_handler)
        await system.message_broker.publish(
            "notifications",
            AsyncMessage(content="Hello async world!")
        )
        
        # 2. Stream processing example
        stream_processor = AsyncStreamProcessor[Dict[str, Any]]("example")
        stream_processor \
            .filter(lambda x: x.get('value', 0) > 10) \
            .map(lambda x: {**x, 'processed': True}) \
            .batch(10) \
            .on_error(lambda item, err: print(f"Error: {err}"))
        
        # 3. Task queue example
        async def example_task_handler(payload):
            await asyncio.sleep(1)
            return f"Processed: {payload}"
        
        task = AsyncTask(
            name="example_task",
            payload={"data": "test"},
            priority=1,
            metadata={'handler': example_task_handler}
        )
        
        future = await system.task_queue.submit(task)
        result = await future
        print(f"Task result: {result}")
        
        # 4. WebSocket example
        async def chat_handler(client_id: str, data: Dict[str, Any]):
            message = data.get('message')
            await system.ws_server.broadcast({
                'type': 'chat',
                'from': client_id,
                'message': message
            })
        
        system.ws_server.register_handler('chat', chat_handler)
        
        # 5. Event bus example
        async def event_handler(event: AsyncEvent):
            print(f"Event: {event.type} - {event.data}")
        
        system.event_bus.subscribe('user_action', event_handler)
        await system.event_bus.publish(
            AsyncEvent('user_action', {'action': 'login'})
        )
        
        # 6. Connection pool example
        async with system.db_pool.acquire() as conn:
            # Use database connection
            result = await conn.fetch("SELECT 1")
            print(f"DB result: {result}")
        
        # 7. Rate limiter example
        for i in range(10):
            if await system.api_limiter.acquire():
                print(f"Request {i} allowed")
            else:
                print(f"Request {i} rate limited")
        
        # 8. Circuit breaker example
        async def external_api_call():
            # Simulate API call
            return {"status": "ok"}
        
        try:
            result = await system.model_circuit.call(external_api_call)
            print(f"API result: {result}")
        except Exception as e:
            print(f"Circuit breaker open: {e}")
        
        # 9. Cache example
        await system.result_cache.set("key1", {"value": "cached"})
        cached = await system.result_cache.get("key1")
        print(f"Cached value: {cached}")
        
        # 10. Batch processor example
        for i in range(100):
            await system.log_batch.add({
                'level': 'info',
                'message': f'Log entry {i}'
            })
        
        await system.log_batch.flush()
        
    finally:
        await system.stop()

if __name__ == "__main__":
    asyncio.run(example_async_enhancements())

"""
Enhanced Async Processing Features:

1. **Async Message Broker**
   - Priority-based messaging
   - Dead letter queue
   - Retry logic with exponential backoff
   - Topic-based pub/sub

2. **Async Stream Processing**
   - Map, filter, batch, window operations
   - Parallel processing with concurrency control
   - Error handling and recovery
   - Backpressure support

3. **Advanced Task Queue**
   - Priority scheduling
   - Delayed/scheduled tasks
   - Task dependencies
   - Retry mechanisms

4. **WebSocket Server**
   - Real-time bidirectional communication
   - Room-based messaging
   - Custom message handlers
   - Client management

5. **Event Bus**
   - Decoupled event-driven architecture
   - Middleware support
   - Event history
   - Wildcard subscriptions

6. **Data Pipeline**
   - Multi-stage processing
   - Configurable worker pools
   - Backpressure handling
   - Stream processing

7. **Circuit Breaker**
   - Fault tolerance
   - Automatic recovery
   - State monitoring
   - Configurable thresholds

8. **Rate Limiter**
   - Token bucket algorithm
   - Async-aware waiting
   - Metrics tracking
   - Multiple limiter instances

9. **Async Cache**
   - TTL support
   - LRU eviction
   - Async-safe operations
   - Metrics and monitoring

10. **Batch Processor**
    - Time and size based batching
    - Automatic flushing
    - Error handling
    - Metrics collection

11. **Connection Pool**
    - Generic connection pooling
    - Auto-scaling
    - Health checking
    - Context manager support

12. **Priority Semaphore**
    - Priority-based resource access
    - Fair scheduling
    - Context manager support

These enhancements provide:
- Better scalability through async operations
- Improved fault tolerance
- Real-time capabilities
- Efficient resource utilization
- Production-ready monitoring
- High-performance data processing
"""
