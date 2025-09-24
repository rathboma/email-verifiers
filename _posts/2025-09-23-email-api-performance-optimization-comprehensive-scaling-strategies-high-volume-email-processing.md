---
layout: post
title: "Email API Performance Optimization: Comprehensive Scaling Strategies for High-Volume Email Processing"
date: 2025-09-23 08:00:00 -0500
categories: email-api performance-optimization scaling high-volume rate-limiting caching database-optimization
excerpt: "Master email API performance optimization with advanced scaling strategies, caching systems, and database optimization techniques. Learn to build high-performance email processing infrastructure that handles millions of requests efficiently while maintaining reliability and cost-effectiveness."
---

# Email API Performance Optimization: Comprehensive Scaling Strategies for High-Volume Email Processing

Email API performance optimization has become critical as organizations scale their email marketing operations to handle millions of messages across complex, multi-channel campaigns. Modern email systems require sophisticated optimization strategies that balance throughput, reliability, and cost-effectiveness while maintaining exceptional user experiences and meeting strict performance requirements across diverse use cases.

Organizations implementing comprehensive email API optimization strategies typically achieve 70-85% improvements in processing speed, 60-75% reduction in infrastructure costs, and 90-95% reliability improvements compared to unoptimized systems. These performance gains result from systematic optimization across all layers of the email processing stack including database design, caching strategies, queue management, and infrastructure scaling.

This comprehensive guide explores advanced email API optimization techniques, scaling architectures, and performance monitoring systems that enable engineering teams, platform architects, and technical leaders to build email infrastructure capable of processing millions of emails efficiently while maintaining high availability and optimal resource utilization across all operational scenarios.

## Understanding Email API Performance Architecture

### Core Performance Components

Email API performance depends on optimized interaction between multiple system components:

**Request Processing Layer:**
- **API Gateway**: Request routing, rate limiting, and authentication
- **Load Balancer**: Traffic distribution and health monitoring
- **Application Servers**: Business logic processing and validation
- **Message Queue**: Asynchronous processing and task distribution

**Data Processing Layer:**
- **Database Systems**: Contact management and campaign data
- **Cache Layers**: Hot data storage and query optimization
- **File Storage**: Template storage and asset management
- **Analytics Systems**: Real-time metrics and performance tracking

**Email Delivery Layer:**
- **SMTP Servers**: Message transmission and delivery
- **Webhook Processors**: Event handling and status updates
- **Monitoring Systems**: Deliverability tracking and error handling
- **Retry Mechanisms**: Failed delivery recovery and optimization

### High-Performance Email API Architecture

Build scalable email API systems optimized for high-volume processing:

{% raw %}
```python
# High-performance email API system with comprehensive optimization
import asyncio
import aiohttp
import aioredis
import aiomysql
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from concurrent.futures import ThreadPoolExecutor
import uvloop
import orjson
import msgpack
import lz4.frame
import numpy as np
from cachetools import TTLCache
import aioboto3
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# Performance monitoring and metrics
email_requests_total = Counter('email_requests_total', 'Total email API requests', ['endpoint', 'status'])
email_processing_duration = Histogram('email_processing_duration_seconds', 'Email processing time')
email_queue_size = Gauge('email_queue_size', 'Current email queue size')
database_connections = Gauge('database_connections_active', 'Active database connections')
cache_hits_total = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
cache_misses_total = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])

class EmailPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ProcessingStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class EmailJob:
    job_id: str
    recipient: str
    template_id: str
    variables: Dict[str, Any]
    priority: EmailPriority
    scheduled_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    status: ProcessingStatus = ProcessingStatus.QUEUED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'recipient': self.recipient,
            'template_id': self.template_id,
            'variables': self.variables,
            'priority': self.priority.value,
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailJob':
        return cls(
            job_id=data['job_id'],
            recipient=data['recipient'],
            template_id=data['template_id'],
            variables=data['variables'],
            priority=EmailPriority(data['priority']),
            scheduled_time=datetime.fromisoformat(data['scheduled_time']) if data['scheduled_time'] else None,
            retry_count=data['retry_count'],
            max_retries=data['max_retries'],
            created_at=datetime.fromisoformat(data['created_at']),
            status=ProcessingStatus(data['status'])
        )

class HighPerformanceCache:
    def __init__(self, redis_url: str, max_memory_cache_size: int = 10000):
        self.redis_url = redis_url
        self.redis_pool = None
        self.memory_cache = TTLCache(maxsize=max_memory_cache_size, ttl=300)  # 5-minute TTL
        self.compression_threshold = 1024  # Compress values larger than 1KB
        
    async def initialize(self):
        """Initialize Redis connection pool"""
        self.redis_pool = aioredis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=100,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={1: 1, 2: 3, 3: 5}
        )
    
    async def get(self, key: str, cache_type: str = "redis") -> Optional[Any]:
        """Get value from cache with multi-tier fallback"""
        # Try memory cache first
        if key in self.memory_cache:
            cache_hits_total.labels(cache_type="memory").inc()
            return self.memory_cache[key]
        
        # Try Redis cache
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            raw_value = await redis.get(key)
            
            if raw_value:
                cache_hits_total.labels(cache_type="redis").inc()
                
                # Decompress if needed
                if raw_value.startswith(b'lz4:'):
                    raw_value = lz4.frame.decompress(raw_value[4:])
                
                # Deserialize
                value = orjson.loads(raw_value)
                
                # Store in memory cache for faster access
                self.memory_cache[key] = value
                return value
            else:
                cache_misses_total.labels(cache_type=cache_type).inc()
                return None
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            cache_misses_total.labels(cache_type=cache_type).inc()
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with compression and multi-tier storage"""
        try:
            # Store in memory cache
            self.memory_cache[key] = value
            
            # Serialize value
            serialized = orjson.dumps(value)
            
            # Compress if over threshold
            if len(serialized) > self.compression_threshold:
                compressed = lz4.frame.compress(serialized)
                if len(compressed) < len(serialized):
                    serialized = b'lz4:' + compressed
            
            # Store in Redis
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            await redis.setex(key, ttl, serialized)
            
            return True
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Remove from Redis
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            await redis.delete(key)
            
            return True
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False

class DatabaseConnectionManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.read_pool = None
        self.write_pool = None
        self.connection_semaphore = asyncio.Semaphore(config.get('max_connections', 50))
        
    async def initialize(self):
        """Initialize database connection pools"""
        # Write pool (primary database)
        self.write_pool = await asyncpg.create_pool(
            host=self.config['write_host'],
            port=self.config['port'],
            user=self.config['user'],
            password=self.config['password'],
            database=self.config['database'],
            min_size=10,
            max_size=30,
            command_timeout=60,
            server_settings={
                'application_name': 'email-api-writer',
                'tcp_keepalives_idle': '600',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3'
            }
        )
        
        # Read pool (read replica if available)
        read_host = self.config.get('read_host', self.config['write_host'])
        self.read_pool = await asyncpg.create_pool(
            host=read_host,
            port=self.config['port'],
            user=self.config['user'],
            password=self.config['password'],
            database=self.config['database'],
            min_size=20,
            max_size=50,
            command_timeout=30,
            server_settings={
                'application_name': 'email-api-reader',
                'tcp_keepalives_idle': '600',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3'
            }
        )
    
    @asynccontextmanager
    async def get_connection(self, read_only: bool = False):
        """Get database connection with automatic cleanup"""
        async with self.connection_semaphore:
            pool = self.read_pool if read_only else self.write_pool
            async with pool.acquire() as connection:
                database_connections.inc()
                try:
                    yield connection
                finally:
                    database_connections.dec()

class OptimizedEmailQueue:
    def __init__(self, redis_url: str, max_concurrent_jobs: int = 1000):
        self.redis_url = redis_url
        self.redis_pool = None
        self.max_concurrent_jobs = max_concurrent_jobs
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self.priority_queues = {
            EmailPriority.CRITICAL: "email_queue:critical",
            EmailPriority.HIGH: "email_queue:high",
            EmailPriority.NORMAL: "email_queue:normal",
            EmailPriority.LOW: "email_queue:low"
        }
        
    async def initialize(self):
        """Initialize Redis connection for queue"""
        self.redis_pool = aioredis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=50,
            retry_on_timeout=True
        )
    
    async def enqueue(self, job: EmailJob) -> bool:
        """Add job to priority queue"""
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            queue_name = self.priority_queues[job.priority]
            
            # Serialize job
            job_data = msgpack.packb(job.to_dict())
            
            # Add to queue with score based on priority and timestamp
            score = time.time() + (4 - job.priority.value) * 1000
            
            await redis.zadd(queue_name, {job_data: score})
            email_queue_size.inc()
            
            return True
        except Exception as e:
            logging.error(f"Queue enqueue error: {e}")
            return False
    
    async def dequeue(self) -> Optional[EmailJob]:
        """Get next job from highest priority queue"""
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Check queues in priority order
            for priority in [EmailPriority.CRITICAL, EmailPriority.HIGH, EmailPriority.NORMAL, EmailPriority.LOW]:
                queue_name = self.priority_queues[priority]
                
                # Get job with lowest score (oldest with highest priority)
                result = await redis.zpopmin(queue_name, count=1)
                
                if result:
                    job_data = msgpack.unpackb(result[0][0])
                    email_queue_size.dec()
                    return EmailJob.from_dict(job_data)
            
            return None
        except Exception as e:
            logging.error(f"Queue dequeue error: {e}")
            return None
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics"""
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            stats = {}
            
            for priority, queue_name in self.priority_queues.items():
                count = await redis.zcard(queue_name)
                stats[priority.name.lower()] = count
            
            return stats
        except Exception as e:
            logging.error(f"Queue stats error: {e}")
            return {}

class EmailProcessingEngine:
    def __init__(self, 
                 cache: HighPerformanceCache,
                 db_manager: DatabaseConnectionManager,
                 queue: OptimizedEmailQueue):
        self.cache = cache
        self.db_manager = db_manager
        self.queue = queue
        self.processing_workers = []
        self.template_cache = {}
        self.rate_limiter = {}
        
    async def start_workers(self, num_workers: int = 10):
        """Start background processing workers"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._processing_worker(i))
            self.processing_workers.append(worker)
    
    async def _processing_worker(self, worker_id: int):
        """Background worker for processing email jobs"""
        logging.info(f"Starting email processing worker {worker_id}")
        
        while True:
            try:
                # Get next job from queue
                job = await self.queue.dequeue()
                
                if job:
                    async with self.queue.processing_semaphore:
                        await self._process_email_job(job, worker_id)
                else:
                    # No jobs available, wait before checking again
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _process_email_job(self, job: EmailJob, worker_id: int):
        """Process individual email job"""
        start_time = time.time()
        
        try:
            # Update job status
            job.status = ProcessingStatus.PROCESSING
            
            # Check rate limits
            if not await self._check_rate_limits(job.recipient):
                # Re-queue with delay
                job.scheduled_time = datetime.now() + timedelta(minutes=5)
                await self.queue.enqueue(job)
                return
            
            # Get and render template
            template_content = await self._get_template(job.template_id)
            if not template_content:
                raise Exception(f"Template {job.template_id} not found")
            
            rendered_content = await self._render_template(template_content, job.variables)
            
            # Send email
            success = await self._send_email(job.recipient, rendered_content)
            
            if success:
                job.status = ProcessingStatus.SENT
                await self._record_sent_email(job)
            else:
                raise Exception("Email sending failed")
                
        except Exception as e:
            logging.error(f"Email processing error: {e}")
            job.retry_count += 1
            
            if job.retry_count <= job.max_retries:
                job.status = ProcessingStatus.RETRYING
                job.scheduled_time = datetime.now() + timedelta(minutes=2 ** job.retry_count)
                await self.queue.enqueue(job)
            else:
                job.status = ProcessingStatus.FAILED
                await self._record_failed_email(job, str(e))
        
        finally:
            processing_time = time.time() - start_time
            email_processing_duration.observe(processing_time)
    
    async def _get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get email template with caching"""
        cache_key = f"template:{template_id}"
        
        # Try cache first
        template = await self.cache.get(cache_key, "template")
        if template:
            return template
        
        # Load from database
        async with self.db_manager.get_connection(read_only=True) as conn:
            row = await conn.fetchrow(
                "SELECT template_id, subject, html_content, text_content, variables "
                "FROM email_templates WHERE template_id = $1 AND active = true",
                template_id
            )
            
            if row:
                template = dict(row)
                # Cache for 1 hour
                await self.cache.set(cache_key, template, ttl=3600)
                return template
        
        return None
    
    async def _render_template(self, template: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, str]:
        """Render email template with variables"""
        # Simple template rendering - in production, use proper template engine
        subject = template['subject']
        html_content = template['html_content']
        text_content = template['text_content']
        
        # Replace variables
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            subject = subject.replace(placeholder, str(value))
            html_content = html_content.replace(placeholder, str(value))
            text_content = text_content.replace(placeholder, str(value))
        
        return {
            'subject': subject,
            'html_content': html_content,
            'text_content': text_content
        }
    
    async def _check_rate_limits(self, recipient: str) -> bool:
        """Check if recipient is within rate limits"""
        # Implement domain-based rate limiting
        domain = recipient.split('@')[1]
        rate_key = f"rate_limit:{domain}"
        
        current_time = time.time()
        
        # Get current rate limit data
        rate_data = await self.cache.get(rate_key, "rate_limit")
        
        if not rate_data:
            rate_data = {'count': 0, 'window_start': current_time}
        
        # Reset window if needed (1-minute windows)
        if current_time - rate_data['window_start'] > 60:
            rate_data = {'count': 0, 'window_start': current_time}
        
        # Check domain-specific limits
        domain_limits = {
            'gmail.com': 100,    # 100 emails per minute
            'yahoo.com': 50,     # 50 emails per minute
            'outlook.com': 75,   # 75 emails per minute
            'hotmail.com': 75,   # 75 emails per minute
        }
        
        limit = domain_limits.get(domain, 200)  # Default: 200 per minute
        
        if rate_data['count'] >= limit:
            return False
        
        # Increment counter and update cache
        rate_data['count'] += 1
        await self.cache.set(rate_key, rate_data, ttl=120)
        
        return True
    
    async def _send_email(self, recipient: str, content: Dict[str, str]) -> bool:
        """Send email via SMTP or email service"""
        # Simulate email sending - replace with actual SMTP/service integration
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate 95% success rate
        return np.random.random() > 0.05
    
    async def _record_sent_email(self, job: EmailJob):
        """Record successful email delivery"""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO email_logs (job_id, recipient, template_id, status, sent_at, processing_time)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                job.job_id,
                job.recipient,
                job.template_id,
                job.status.value,
                datetime.now(),
                (datetime.now() - job.created_at).total_seconds()
            )
    
    async def _record_failed_email(self, job: EmailJob, error_message: str):
        """Record failed email delivery"""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO email_logs (job_id, recipient, template_id, status, error_message, failed_at, retry_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                job.job_id,
                job.recipient,
                job.template_id,
                job.status.value,
                error_message,
                datetime.now(),
                job.retry_count
            )

class EmailAPIServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="High-Performance Email API")
        self.cache = None
        self.db_manager = None
        self.queue = None
        self.processing_engine = None
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v1/emails/send")
        async def send_email(email_data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Send single email"""
            try:
                email_requests_total.labels(endpoint="send", status="received").inc()
                
                # Validate request
                if not all(key in email_data for key in ['recipient', 'template_id']):
                    email_requests_total.labels(endpoint="send", status="invalid").inc()
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                # Create email job
                job = EmailJob(
                    job_id=str(uuid.uuid4()),
                    recipient=email_data['recipient'],
                    template_id=email_data['template_id'],
                    variables=email_data.get('variables', {}),
                    priority=EmailPriority(email_data.get('priority', 2))
                )
                
                # Queue email
                success = await self.queue.enqueue(job)
                
                if success:
                    email_requests_total.labels(endpoint="send", status="queued").inc()
                    return {"job_id": job.job_id, "status": "queued"}
                else:
                    email_requests_total.labels(endpoint="send", status="error").inc()
                    raise HTTPException(status_code=500, detail="Failed to queue email")
                    
            except Exception as e:
                email_requests_total.labels(endpoint="send", status="error").inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/emails/send-batch")
        async def send_batch_emails(batch_data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Send batch of emails"""
            try:
                email_requests_total.labels(endpoint="send_batch", status="received").inc()
                
                emails = batch_data.get('emails', [])
                if not emails:
                    email_requests_total.labels(endpoint="send_batch", status="invalid").inc()
                    raise HTTPException(status_code=400, detail="No emails provided")
                
                job_ids = []
                
                # Process each email in batch
                for email_data in emails:
                    if not all(key in email_data for key in ['recipient', 'template_id']):
                        continue
                    
                    job = EmailJob(
                        job_id=str(uuid.uuid4()),
                        recipient=email_data['recipient'],
                        template_id=email_data['template_id'],
                        variables=email_data.get('variables', {}),
                        priority=EmailPriority(email_data.get('priority', 2))
                    )
                    
                    success = await self.queue.enqueue(job)
                    if success:
                        job_ids.append(job.job_id)
                
                email_requests_total.labels(endpoint="send_batch", status="queued").inc()
                return {"job_ids": job_ids, "total_queued": len(job_ids)}
                
            except Exception as e:
                email_requests_total.labels(endpoint="send_batch", status="error").inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/emails/{job_id}/status")
        async def get_email_status(job_id: str):
            """Get email job status"""
            try:
                email_requests_total.labels(endpoint="status", status="received").inc()
                
                async with self.db_manager.get_connection(read_only=True) as conn:
                    row = await conn.fetchrow(
                        "SELECT job_id, recipient, template_id, status, sent_at, failed_at, error_message "
                        "FROM email_logs WHERE job_id = $1 ORDER BY created_at DESC LIMIT 1",
                        job_id
                    )
                    
                    if row:
                        email_requests_total.labels(endpoint="status", status="found").inc()
                        return dict(row)
                    else:
                        email_requests_total.labels(endpoint="status", status="not_found").inc()
                        raise HTTPException(status_code=404, detail="Job not found")
                        
            except HTTPException:
                raise
            except Exception as e:
                email_requests_total.labels(endpoint="status", status="error").inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Get API metrics"""
            try:
                queue_stats = await self.queue.get_queue_stats()
                
                return {
                    "queue_stats": queue_stats,
                    "total_queue_size": sum(queue_stats.values()),
                    "cache_stats": {
                        "memory_cache_size": len(self.cache.memory_cache),
                        "memory_cache_max_size": self.cache.memory_cache.maxsize
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    async def initialize(self):
        """Initialize all system components"""
        # Initialize cache
        self.cache = HighPerformanceCache(self.config['redis_url'])
        await self.cache.initialize()
        
        # Initialize database
        self.db_manager = DatabaseConnectionManager(self.config['database'])
        await self.db_manager.initialize()
        
        # Initialize queue
        self.queue = OptimizedEmailQueue(self.config['redis_url'])
        await self.queue.initialize()
        
        # Initialize processing engine
        self.processing_engine = EmailProcessingEngine(self.cache, self.db_manager, self.queue)
        await self.processing_engine.start_workers(self.config.get('num_workers', 10))
        
        # Start Prometheus metrics server
        start_http_server(self.config.get('metrics_port', 8080))
    
    async def shutdown(self):
        """Graceful shutdown"""
        # Stop processing workers
        for worker in self.processing_engine.processing_workers:
            worker.cancel()
        
        # Close database connections
        if self.db_manager.write_pool:
            await self.db_manager.write_pool.close()
        if self.db_manager.read_pool:
            await self.db_manager.read_pool.close()
        
        # Close Redis connections
        if self.cache.redis_pool:
            await self.cache.redis_pool.disconnect()

# Advanced performance monitoring and optimization
class PerformanceOptimizer:
    def __init__(self, api_server: EmailAPIServer):
        self.api_server = api_server
        self.performance_history = []
        self.optimization_rules = {}
        
    async def monitor_performance(self):
        """Continuously monitor and optimize performance"""
        while True:
            try:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 100 measurements
                if len(self.performance_history) > 100:
                    self.performance_history.pop(0)
                
                # Analyze and optimize
                await self._analyze_and_optimize(metrics)
                
                # Wait 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics"""
        queue_stats = await self.api_server.queue.get_queue_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'queue_size': sum(queue_stats.values()),
            'queue_breakdown': queue_stats,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'avg_processing_time': self._get_avg_processing_time(),
            'active_connections': database_connections._value.get(),
            'memory_usage': self._get_memory_usage()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        # This would integrate with actual metrics
        return 0.85  # Placeholder
    
    def _get_avg_processing_time(self) -> float:
        """Get average email processing time"""
        # This would integrate with actual metrics
        return 0.5  # Placeholder
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        # This would integrate with actual system monitoring
        return {"heap_used": 512.0, "heap_total": 1024.0}  # Placeholder
    
    async def _analyze_and_optimize(self, current_metrics: Dict[str, Any]):
        """Analyze metrics and apply optimizations"""
        # Queue size optimization
        if current_metrics['queue_size'] > 10000:
            await self._optimize_queue_processing()
        
        # Cache optimization
        if current_metrics['cache_hit_rate'] < 0.7:
            await self._optimize_cache_strategy()
        
        # Database connection optimization
        if current_metrics['active_connections'] > 40:
            await self._optimize_database_connections()
    
    async def _optimize_queue_processing(self):
        """Optimize queue processing when overloaded"""
        # Increase worker count temporarily
        current_workers = len(self.api_server.processing_engine.processing_workers)
        if current_workers < 20:
            await self.api_server.processing_engine.start_workers(5)
            logging.info("Added 5 additional processing workers due to queue overload")
    
    async def _optimize_cache_strategy(self):
        """Optimize caching strategy for better hit rates"""
        # Increase cache TTL for templates
        logging.info("Optimizing cache strategy - increasing template cache TTL")
    
    async def _optimize_database_connections(self):
        """Optimize database connection usage"""
        # This would implement connection pooling adjustments
        logging.info("Optimizing database connections")

# Configuration and startup system
class EmailAPIConfiguration:
    def __init__(self):
        self.config = {
            'database': {
                'write_host': 'localhost',
                'read_host': 'localhost',  # Can be read replica
                'port': 5432,
                'user': 'email_api',
                'password': 'secure_password',
                'database': 'email_system',
                'max_connections': 50
            },
            'redis_url': 'redis://localhost:6379/0',
            'num_workers': 10,
            'metrics_port': 8080,
            'api_port': 8000,
            'log_level': 'INFO',
            'enable_monitoring': True
        }
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        import os
        
        # Database configuration
        if os.getenv('DB_HOST'):
            self.config['database']['write_host'] = os.getenv('DB_HOST')
        if os.getenv('DB_READ_HOST'):
            self.config['database']['read_host'] = os.getenv('DB_READ_HOST')
        if os.getenv('DB_USER'):
            self.config['database']['user'] = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self.config['database']['password'] = os.getenv('DB_PASSWORD')
        if os.getenv('DB_NAME'):
            self.config['database']['database'] = os.getenv('DB_NAME')
        
        # Redis configuration
        if os.getenv('REDIS_URL'):
            self.config['redis_url'] = os.getenv('REDIS_URL')
        
        # Performance configuration
        if os.getenv('NUM_WORKERS'):
            self.config['num_workers'] = int(os.getenv('NUM_WORKERS'))
        if os.getenv('API_PORT'):
            self.config['api_port'] = int(os.getenv('API_PORT'))
        if os.getenv('METRICS_PORT'):
            self.config['metrics_port'] = int(os.getenv('METRICS_PORT'))
        
        return self.config

# Database schema setup
SCHEMA_SQL = """
-- Email templates table
CREATE TABLE IF NOT EXISTS email_templates (
    template_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    subject TEXT NOT NULL,
    html_content TEXT NOT NULL,
    text_content TEXT NOT NULL,
    variables JSONB,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_email_templates_active ON email_templates(active);

-- Email logs table
CREATE TABLE IF NOT EXISTS email_logs (
    id BIGSERIAL PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL,
    recipient VARCHAR(255) NOT NULL,
    template_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    sent_at TIMESTAMP,
    failed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processing_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_email_logs_job_id ON email_logs(job_id);
CREATE INDEX idx_email_logs_recipient ON email_logs(recipient);
CREATE INDEX idx_email_logs_status ON email_logs(status);
CREATE INDEX idx_email_logs_created_at ON email_logs(created_at DESC);

-- Performance optimization indexes
CREATE INDEX idx_email_logs_status_created ON email_logs(status, created_at DESC);
CREATE INDEX idx_email_logs_template_status ON email_logs(template_id, status);
"""

# Main application startup
async def create_email_api_server():
    """Create and initialize email API server"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Load configuration
    config_manager = EmailAPIConfiguration()
    config = config_manager.load_from_env()
    
    # Initialize API server
    api_server = EmailAPIServer(config)
    await api_server.initialize()
    
    # Start performance optimizer
    if config['enable_monitoring']:
        optimizer = PerformanceOptimizer(api_server)
        asyncio.create_task(optimizer.monitor_performance())
    
    return api_server

# Usage example and demonstration
async def demonstrate_high_performance_email_api():
    """
    Demonstrate high-performance email API system
    """
    
    print("=== High-Performance Email API Demo ===")
    
    # Create and initialize server
    api_server = await create_email_api_server()
    
    print("✓ Email API server initialized")
    print("✓ Database connections established")
    print("✓ Redis cache initialized")
    print("✓ Processing workers started")
    print("✓ Performance monitoring active")
    
    # Simulate some email jobs
    sample_jobs = [
        EmailJob(
            job_id=str(uuid.uuid4()),
            recipient=f"user{i}@example.com",
            template_id="welcome_email",
            variables={"name": f"User {i}", "company": "Demo Corp"},
            priority=EmailPriority.NORMAL if i % 2 == 0 else EmailPriority.HIGH
        )
        for i in range(100)
    ]
    
    # Queue jobs
    queued_count = 0
    for job in sample_jobs:
        success = await api_server.queue.enqueue(job)
        if success:
            queued_count += 1
    
    print(f"✓ Queued {queued_count} sample email jobs")
    
    # Wait for some processing
    await asyncio.sleep(5)
    
    # Get queue statistics
    queue_stats = await api_server.queue.get_queue_stats()
    print(f"Queue statistics: {queue_stats}")
    
    # Demonstrate batch processing
    batch_data = {
        'emails': [
            {
                'recipient': f'batch{i}@example.com',
                'template_id': 'newsletter',
                'variables': {'issue_number': 42, 'subscriber_name': f'Subscriber {i}'},
                'priority': 2
            }
            for i in range(50)
        ]
    }
    
    print("\n--- Performance Metrics ---")
    print(f"Active database connections: {database_connections._value.get()}")
    print(f"Total email requests processed: {email_requests_total._value.sum()}")
    print(f"Average processing time: {email_processing_duration._value.sum():.2f}s")
    print(f"Cache hit rate: ~85%")  # Estimated
    
    return {
        'server_initialized': True,
        'jobs_queued': queued_count,
        'workers_active': len(api_server.processing_engine.processing_workers),
        'performance_optimized': True
    }

if __name__ == "__main__":
    # Use uvloop for better performance
    uvloop.install()
    
    async def run_demo():
        result = await demonstrate_high_performance_email_api()
        
        print(f"\n=== Email API Demo Complete ===")
        print(f"Server initialized: {result['server_initialized']}")
        print(f"Jobs queued: {result['jobs_queued']}")
        print(f"Active workers: {result['workers_active']}")
        print("High-performance email API system operational")
        print("Ready for production email processing at scale")
    
    asyncio.run(run_demo())
```
{% endraw %}

## Advanced Caching Strategies for Email APIs

### Multi-Tier Cache Architecture

Implement sophisticated caching systems that dramatically improve email API performance:

**Memory Cache Layer:**
- Hot data storage with sub-millisecond access times
- Template fragments and frequently accessed configurations
- User session data and authentication tokens
- Recent email job status information

**Distributed Cache Layer:**
- Redis cluster for shared cache across multiple API instances
- Compressed data storage for large objects
- Cross-instance cache invalidation and synchronization
- Rate limiting counters and distributed locks

**Database Query Cache:**
- Query result caching with intelligent invalidation
- Materialized views for complex aggregations
- Connection pooling and prepared statement caching
- Read replica routing for scalability

### Intelligent Cache Management

{% raw %}
```python
# Advanced cache management system for email APIs
import pickle
import zlib
from typing import Optional, Any, Dict, List, Callable
from datetime import datetime, timedelta
import asyncio
import weakref
from collections import defaultdict, OrderedDict
import threading
from dataclasses import dataclass
import hashlib
import json
from enum import Enum

class CacheLevel(Enum):
    L1_MEMORY = 1
    L2_REDIS = 2
    L3_DATABASE = 3

class CacheStrategy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int]
    size_bytes: int
    compression_ratio: float = 1.0
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def calculate_priority(self, strategy: CacheStrategy) -> float:
        """Calculate cache entry priority for eviction"""
        now = datetime.now()
        age_seconds = (now - self.created_at).total_seconds()
        last_access_seconds = (now - self.last_accessed).total_seconds()
        
        if strategy == CacheStrategy.LRU:
            return -last_access_seconds  # More recently used = higher priority
        elif strategy == CacheStrategy.LFU:
            return self.access_count / max(age_seconds, 1)  # Normalize by age
        elif strategy == CacheStrategy.TTL:
            if self.ttl is None:
                return float('inf')
            return self.ttl - age_seconds
        else:
            return 0.0

class IntelligentCacheManager:
    def __init__(self, 
                 max_memory_size: int = 100 * 1024 * 1024,  # 100MB
                 eviction_strategy: CacheStrategy = CacheStrategy.LRU,
                 compression_threshold: int = 1024):  # 1KB
        
        self.max_memory_size = max_memory_size
        self.eviction_strategy = eviction_strategy
        self.compression_threshold = compression_threshold
        
        # Multi-level storage
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l2_cache = None  # Redis connection
        self.l3_cache = None  # Database connection
        
        # Management structures
        self.current_size = 0
        self.access_lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'decompressions': 0
        }
        
        # Background maintenance
        self._maintenance_task = None
        self._stop_maintenance = False
    
    async def initialize_l2_cache(self, redis_connection):
        """Initialize L2 Redis cache"""
        self.l2_cache = redis_connection
    
    async def initialize_l3_cache(self, db_connection):
        """Initialize L3 database cache"""
        self.l3_cache = db_connection
    
    async def get(self, key: str, cache_levels: List[CacheLevel] = None) -> Optional[Any]:
        """Get value from cache with multi-level fallback"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]
        
        for level in cache_levels:
            try:
                if level == CacheLevel.L1_MEMORY:
                    value = await self._get_from_l1(key)
                elif level == CacheLevel.L2_REDIS:
                    value = await self._get_from_l2(key)
                elif level == CacheLevel.L3_DATABASE:
                    value = await self._get_from_l3(key)
                else:
                    continue
                
                if value is not None:
                    # Promote to higher cache levels
                    await self._promote_to_higher_levels(key, value, level, cache_levels)
                    self.stats['hits'] += 1
                    return value
                    
            except Exception as e:
                logging.error(f"Cache get error at level {level}: {e}")
                continue
        
        self.stats['misses'] += 1
        return None
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get value from L1 memory cache"""
        with self.access_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.l1_cache[key]
                    self.current_size -= entry.size_bytes
                    return None
                
                # Update access information
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Decompress if needed
                value = entry.value
                if isinstance(value, bytes) and value.startswith(b'compressed:'):
                    value = pickle.loads(zlib.decompress(value[11:]))
                    self.stats['decompressions'] += 1
                
                return value
        
        return None
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 Redis cache"""
        if not self.l2_cache:
            return None
        
        try:
            raw_value = await self.l2_cache.get(key)
            if raw_value:
                # Try to deserialize
                if raw_value.startswith(b'json:'):
                    return json.loads(raw_value[5:].decode())
                elif raw_value.startswith(b'pickle:'):
                    return pickle.loads(raw_value[7:])
                elif raw_value.startswith(b'compressed:'):
                    return pickle.loads(zlib.decompress(raw_value[11:]))
                else:
                    return raw_value.decode()
        except Exception as e:
            logging.error(f"L2 cache deserialization error: {e}")
        
        return None
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 database cache"""
        if not self.l3_cache:
            return None
        
        try:
            # This would query the database cache table
            # Implementation depends on specific database schema
            row = await self.l3_cache.fetchrow(
                "SELECT cache_value, created_at, ttl FROM cache_table WHERE cache_key = $1",
                key
            )
            
            if row:
                # Check TTL
                if row['ttl'] and row['created_at'] + timedelta(seconds=row['ttl']) < datetime.now():
                    # Expired, cleanup
                    await self.l3_cache.execute(
                        "DELETE FROM cache_table WHERE cache_key = $1", key
                    )
                    return None
                
                # Deserialize value
                cache_value = row['cache_value']
                if isinstance(cache_value, str):
                    if cache_value.startswith('json:'):
                        return json.loads(cache_value[5:])
                    elif cache_value.startswith('pickle:'):
                        return pickle.loads(bytes.fromhex(cache_value[7:]))
                
                return cache_value
        except Exception as e:
            logging.error(f"L3 cache error: {e}")
        
        return None
    
    async def _promote_to_higher_levels(self, 
                                      key: str, 
                                      value: Any, 
                                      current_level: CacheLevel,
                                      target_levels: List[CacheLevel]):
        """Promote cache entry to higher levels"""
        for level in target_levels:
            if level.value < current_level.value:
                try:
                    if level == CacheLevel.L1_MEMORY:
                        await self.set_l1(key, value)
                    elif level == CacheLevel.L2_REDIS:
                        await self.set_l2(key, value)
                except Exception as e:
                    logging.error(f"Cache promotion error to level {level}: {e}")
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[int] = None,
                  cache_levels: List[CacheLevel] = None) -> bool:
        """Set value in cache across specified levels"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        success = False
        
        for level in cache_levels:
            try:
                if level == CacheLevel.L1_MEMORY:
                    success |= await self.set_l1(key, value, ttl)
                elif level == CacheLevel.L2_REDIS:
                    success |= await self.set_l2(key, value, ttl)
                elif level == CacheLevel.L3_DATABASE:
                    success |= await self.set_l3(key, value, ttl)
            except Exception as e:
                logging.error(f"Cache set error at level {level}: {e}")
        
        return success
    
    async def set_l1(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L1 memory cache"""
        with self.access_lock:
            # Serialize and optionally compress value
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # Compress large values
            compressed_value = serialized
            compression_ratio = 1.0
            
            if size_bytes > self.compression_threshold:
                compressed = zlib.compress(serialized)
                if len(compressed) < size_bytes:
                    compressed_value = b'compressed:' + compressed
                    compression_ratio = len(compressed) / size_bytes
                    size_bytes = len(compressed_value)
                    self.stats['compressions'] += 1
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes,
                compression_ratio=compression_ratio
            )
            
            # Check if we need to evict entries
            while self.current_size + size_bytes > self.max_memory_size and self.l1_cache:
                await self._evict_entry()
            
            # Add to cache
            if key in self.l1_cache:
                self.current_size -= self.l1_cache[key].size_bytes
            
            self.l1_cache[key] = entry
            self.current_size += size_bytes
            
            return True
    
    async def set_l2(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L2 Redis cache"""
        if not self.l2_cache:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = b'json:' + json.dumps(value).encode()
            else:
                serialized = b'pickle:' + pickle.dumps(value)
            
            # Compress if large
            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized)
                if len(compressed) < len(serialized):
                    serialized = b'compressed:' + compressed
            
            # Set with TTL
            if ttl:
                await self.l2_cache.setex(key, ttl, serialized)
            else:
                await self.l2_cache.set(key, serialized)
            
            return True
        except Exception as e:
            logging.error(f"L2 cache set error: {e}")
            return False
    
    async def set_l3(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L3 database cache"""
        if not self.l3_cache:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = 'json:' + json.dumps(value)
            else:
                serialized = 'pickle:' + pickle.dumps(value).hex()
            
            # Insert or update
            await self.l3_cache.execute(
                """
                INSERT INTO cache_table (cache_key, cache_value, created_at, ttl)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (cache_key) DO UPDATE SET
                    cache_value = EXCLUDED.cache_value,
                    created_at = EXCLUDED.created_at,
                    ttl = EXCLUDED.ttl
                """,
                key, serialized, datetime.now(), ttl
            )
            
            return True
        except Exception as e:
            logging.error(f"L3 cache set error: {e}")
            return False
    
    async def _evict_entry(self):
        """Evict entry based on eviction strategy"""
        if not self.l1_cache:
            return
        
        # Calculate priorities for all entries
        entries_with_priority = [
            (key, entry.calculate_priority(self.eviction_strategy), entry)
            for key, entry in self.l1_cache.items()
        ]
        
        # Sort by priority (lowest first for eviction)
        entries_with_priority.sort(key=lambda x: x[1])
        
        # Evict the lowest priority entry
        key_to_evict, _, entry = entries_with_priority[0]
        
        del self.l1_cache[key_to_evict]
        self.current_size -= entry.size_bytes
        self.stats['evictions'] += 1
    
    async def invalidate(self, key: str, cache_levels: List[CacheLevel] = None):
        """Invalidate cache entry across levels"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]
        
        for level in cache_levels:
            try:
                if level == CacheLevel.L1_MEMORY:
                    with self.access_lock:
                        if key in self.l1_cache:
                            entry = self.l1_cache[key]
                            del self.l1_cache[key]
                            self.current_size -= entry.size_bytes
                
                elif level == CacheLevel.L2_REDIS and self.l2_cache:
                    await self.l2_cache.delete(key)
                
                elif level == CacheLevel.L3_DATABASE and self.l3_cache:
                    await self.l3_cache.execute(
                        "DELETE FROM cache_table WHERE cache_key = $1", key
                    )
            except Exception as e:
                logging.error(f"Cache invalidation error at level {level}: {e}")
    
    async def clear(self, cache_levels: List[CacheLevel] = None):
        """Clear cache across levels"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]
        
        for level in cache_levels:
            try:
                if level == CacheLevel.L1_MEMORY:
                    with self.access_lock:
                        self.l1_cache.clear()
                        self.current_size = 0
                
                elif level == CacheLevel.L2_REDIS and self.l2_cache:
                    await self.l2_cache.flushdb()
                
                elif level == CacheLevel.L3_DATABASE and self.l3_cache:
                    await self.l3_cache.execute("TRUNCATE TABLE cache_table")
                    
            except Exception as e:
                logging.error(f"Cache clear error at level {level}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.access_lock:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            
            return {
                'l1_entries': len(self.l1_cache),
                'l1_size_bytes': self.current_size,
                'l1_size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_memory_size / (1024 * 1024),
                'hit_rate': hit_rate,
                'stats': self.stats.copy(),
                'eviction_strategy': self.eviction_strategy.value
            }
    
    async def start_maintenance(self):
        """Start background maintenance task"""
        self._stop_maintenance = False
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def stop_maintenance(self):
        """Stop background maintenance task"""
        self._stop_maintenance = True
        if self._maintenance_task:
            await self._maintenance_task
    
    async def _maintenance_loop(self):
        """Background maintenance for cache optimization"""
        while not self._stop_maintenance:
            try:
                # Clean expired entries
                await self._clean_expired_entries()
                
                # Optimize memory usage
                await self._optimize_memory_usage()
                
                # Update statistics
                await self._update_statistics()
                
                # Wait before next maintenance cycle
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logging.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(60)
    
    async def _clean_expired_entries(self):
        """Clean expired entries from L1 cache"""
        with self.access_lock:
            expired_keys = []
            
            for key, entry in self.l1_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.l1_cache[key]
                del self.l1_cache[key]
                self.current_size -= entry.size_bytes
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage through compression and cleanup"""
        if self.current_size > self.max_memory_size * 0.8:  # 80% threshold
            # More aggressive eviction
            entries_to_evict = max(1, len(self.l1_cache) // 10)  # Evict 10%
            
            for _ in range(entries_to_evict):
                await self._evict_entry()
    
    async def _update_statistics(self):
        """Update cache statistics and metrics"""
        stats = self.get_stats()
        
        # Update Prometheus metrics if available
        if 'prometheus_client' in globals():
            cache_entries_gauge = Gauge('cache_entries_total', 'Total cache entries', ['level'])
            cache_size_gauge = Gauge('cache_size_bytes', 'Cache size in bytes', ['level'])
            cache_hit_rate_gauge = Gauge('cache_hit_rate', 'Cache hit rate')
            
            cache_entries_gauge.labels(level='l1').set(stats['l1_entries'])
            cache_size_gauge.labels(level='l1').set(stats['l1_size_bytes'])
            cache_hit_rate_gauge.set(stats['hit_rate'])
```
{% endraw %}

## Database Optimization for Email APIs

### Query Optimization and Indexing

Implement sophisticated database optimization strategies for high-performance email processing:

{% raw %}
```sql
-- Optimized database schema for high-performance email API
-- Template management with optimized indexing

CREATE TABLE email_templates (
    template_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    subject TEXT NOT NULL,
    html_content TEXT NOT NULL,
    text_content TEXT NOT NULL,
    variables JSONB,
    metadata JSONB,
    active BOOLEAN DEFAULT true,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    -- Indexes for fast lookups
    CONSTRAINT email_templates_name_key UNIQUE (name, version)
);

-- Optimized indexes
CREATE INDEX CONCURRENTLY idx_email_templates_active_category 
    ON email_templates(active, category) WHERE active = true;
CREATE INDEX CONCURRENTLY idx_email_templates_updated_at 
    ON email_templates(updated_at DESC);
CREATE INDEX CONCURRENTLY idx_email_templates_variables_gin 
    ON email_templates USING GIN (variables);

-- Partitioned email logs table for high-volume inserts
CREATE TABLE email_logs (
    id BIGSERIAL,
    job_id VARCHAR(255) NOT NULL,
    recipient VARCHAR(255) NOT NULL,
    template_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    priority INTEGER DEFAULT 2,
    sent_at TIMESTAMP,
    failed_at TIMESTAMP,
    opened_at TIMESTAMP,
    clicked_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processing_time FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE email_logs_y2025m09 PARTITION OF email_logs
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE email_logs_y2025m10 PARTITION OF email_logs
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE email_logs_y2025m11 PARTITION OF email_logs
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- Optimized indexes for each partition
CREATE INDEX CONCURRENTLY idx_email_logs_y2025m09_job_id 
    ON email_logs_y2025m09(job_id);
CREATE INDEX CONCURRENTLY idx_email_logs_y2025m09_recipient_status 
    ON email_logs_y2025m09(recipient, status);
CREATE INDEX CONCURRENTLY idx_email_logs_y2025m09_template_id 
    ON email_logs_y2025m09(template_id);
CREATE INDEX CONCURRENTLY idx_email_logs_y2025m09_status_created 
    ON email_logs_y2025m09(status, created_at DESC);

-- Contact management with optimized lookups
CREATE TABLE contacts (
    contact_id BIGSERIAL PRIMARY KEY,
    email_address VARCHAR(255) NOT NULL,
    email_hash VARCHAR(64) NOT NULL, -- SHA-256 of email for privacy
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    company VARCHAR(255),
    tags TEXT[],
    custom_fields JSONB,
    subscription_status VARCHAR(50) DEFAULT 'subscribed',
    subscription_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_engagement TIMESTAMP,
    engagement_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT contacts_email_unique UNIQUE (email_address)
);

-- Optimized indexes for contact lookups
CREATE INDEX CONCURRENTLY idx_contacts_email_hash 
    ON contacts(email_hash);
CREATE INDEX CONCURRENTLY idx_contacts_subscription_status 
    ON contacts(subscription_status) WHERE subscription_status = 'subscribed';
CREATE INDEX CONCURRENTLY idx_contacts_engagement_score 
    ON contacts(engagement_score DESC) WHERE subscription_status = 'subscribed';
CREATE INDEX CONCURRENTLY idx_contacts_tags_gin 
    ON contacts USING GIN (tags);
CREATE INDEX CONCURRENTLY idx_contacts_custom_fields_gin 
    ON contacts USING GIN (custom_fields);
CREATE INDEX CONCURRENTLY idx_contacts_last_engagement 
    ON contacts(last_engagement DESC) WHERE last_engagement IS NOT NULL;

-- Campaign performance aggregation table
CREATE TABLE campaign_stats (
    campaign_id VARCHAR(255) NOT NULL,
    date_hour TIMESTAMP NOT NULL,
    emails_sent INTEGER DEFAULT 0,
    emails_delivered INTEGER DEFAULT 0,
    emails_opened INTEGER DEFAULT 0,
    emails_clicked INTEGER DEFAULT 0,
    emails_bounced INTEGER DEFAULT 0,
    unsubscribes INTEGER DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0.00,
    processing_time_avg FLOAT DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (campaign_id, date_hour)
);

-- Index for fast aggregations
CREATE INDEX CONCURRENTLY idx_campaign_stats_date_hour 
    ON campaign_stats(date_hour DESC);

-- Database cache table for L3 caching
CREATE TABLE cache_table (
    cache_key VARCHAR(500) PRIMARY KEY,
    cache_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl INTEGER,
    size_bytes INTEGER,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for cache cleanup
CREATE INDEX CONCURRENTLY idx_cache_table_ttl_created 
    ON cache_table(created_at) WHERE ttl IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_cache_table_last_accessed 
    ON cache_table(last_accessed);

-- Database functions for optimized operations

-- Function to update contact engagement score
CREATE OR REPLACE FUNCTION update_contact_engagement_score(
    contact_email VARCHAR(255),
    engagement_type VARCHAR(50),
    engagement_value FLOAT DEFAULT 1.0
) RETURNS VOID AS $$
DECLARE
    score_increment FLOAT;
BEGIN
    -- Calculate score increment based on engagement type
    CASE engagement_type
        WHEN 'open' THEN score_increment := 0.1 * engagement_value;
        WHEN 'click' THEN score_increment := 0.3 * engagement_value;
        WHEN 'conversion' THEN score_increment := 1.0 * engagement_value;
        WHEN 'unsubscribe' THEN score_increment := -2.0;
        ELSE score_increment := 0.0;
    END CASE;
    
    -- Update contact record
    UPDATE contacts 
    SET 
        engagement_score = GREATEST(0, engagement_score + score_increment),
        last_engagement = CASE WHEN engagement_type != 'unsubscribe' 
                              THEN CURRENT_TIMESTAMP 
                              ELSE last_engagement END,
        updated_at = CURRENT_TIMESTAMP
    WHERE email_address = contact_email;
    
    -- Insert into engagement log if needed
    IF score_increment != 0 THEN
        INSERT INTO engagement_log (contact_email, engagement_type, score_change, created_at)
        VALUES (contact_email, engagement_type, score_increment, CURRENT_TIMESTAMP);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get high-performing template recommendations
CREATE OR REPLACE FUNCTION get_template_recommendations(
    category_filter VARCHAR(100) DEFAULT NULL,
    min_performance_score FLOAT DEFAULT 0.7,
    limit_count INTEGER DEFAULT 10
) RETURNS TABLE (
    template_id VARCHAR(255),
    template_name VARCHAR(255),
    performance_score FLOAT,
    total_sends BIGINT,
    avg_open_rate FLOAT,
    avg_click_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH template_performance AS (
        SELECT 
            t.template_id,
            t.name as template_name,
            COUNT(el.id) as total_sends,
            COUNT(CASE WHEN el.opened_at IS NOT NULL THEN 1 END)::FLOAT / 
                NULLIF(COUNT(CASE WHEN el.status = 'sent' THEN 1 END), 0) as open_rate,
            COUNT(CASE WHEN el.clicked_at IS NOT NULL THEN 1 END)::FLOAT / 
                NULLIF(COUNT(CASE WHEN el.status = 'sent' THEN 1 END), 0) as click_rate,
            AVG(el.processing_time) as avg_processing_time
        FROM email_templates t
        LEFT JOIN email_logs el ON t.template_id = el.template_id
        WHERE 
            t.active = true
            AND (category_filter IS NULL OR t.category = category_filter)
            AND el.created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
        GROUP BY t.template_id, t.name
        HAVING COUNT(el.id) > 100  -- Minimum volume for statistical significance
    )
    SELECT 
        tp.template_id,
        tp.template_name,
        -- Performance score: weighted combination of metrics
        (tp.open_rate * 0.4 + tp.click_rate * 0.6) as performance_score,
        tp.total_sends,
        tp.open_rate as avg_open_rate,
        tp.click_rate as avg_click_rate
    FROM template_performance tp
    WHERE (tp.open_rate * 0.4 + tp.click_rate * 0.6) >= min_performance_score
    ORDER BY performance_score DESC, total_sends DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Materialized view for real-time dashboard metrics
CREATE MATERIALIZED VIEW email_dashboard_metrics AS
WITH hourly_stats AS (
    SELECT 
        DATE_TRUNC('hour', created_at) as hour,
        COUNT(*) as total_emails,
        COUNT(CASE WHEN status = 'sent' THEN 1 END) as sent_emails,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_emails,
        COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as opened_emails,
        COUNT(CASE WHEN clicked_at IS NOT NULL THEN 1 END) as clicked_emails,
        AVG(processing_time) as avg_processing_time,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time) as p95_processing_time
    FROM email_logs
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY DATE_TRUNC('hour', created_at)
),
daily_stats AS (
    SELECT 
        DATE_TRUNC('day', created_at) as day,
        COUNT(*) as total_emails,
        COUNT(CASE WHEN status = 'sent' THEN 1 END) as sent_emails,
        COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as opened_emails,
        COUNT(CASE WHEN clicked_at IS NOT NULL THEN 1 END) as clicked_emails,
        COUNT(DISTINCT recipient) as unique_recipients
    FROM email_logs
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', created_at)
)
SELECT 
    'hourly' as metric_type,
    hs.hour as time_period,
    hs.total_emails,
    hs.sent_emails,
    hs.failed_emails,
    hs.opened_emails,
    hs.clicked_emails,
    CASE WHEN hs.sent_emails > 0 
         THEN hs.opened_emails::FLOAT / hs.sent_emails 
         ELSE 0 END as open_rate,
    CASE WHEN hs.sent_emails > 0 
         THEN hs.clicked_emails::FLOAT / hs.sent_emails 
         ELSE 0 END as click_rate,
    hs.avg_processing_time,
    hs.p95_processing_time,
    NULL::BIGINT as unique_recipients
FROM hourly_stats hs
UNION ALL
SELECT 
    'daily' as metric_type,
    ds.day as time_period,
    ds.total_emails,
    ds.sent_emails,
    NULL::BIGINT as failed_emails,
    ds.opened_emails,
    ds.clicked_emails,
    CASE WHEN ds.sent_emails > 0 
         THEN ds.opened_emails::FLOAT / ds.sent_emails 
         ELSE 0 END as open_rate,
    CASE WHEN ds.sent_emails > 0 
         THEN ds.clicked_emails::FLOAT / ds.sent_emails 
         ELSE 0 END as click_rate,
    NULL::FLOAT as avg_processing_time,
    NULL::FLOAT as p95_processing_time,
    ds.unique_recipients
FROM daily_stats ds;

-- Index for fast dashboard queries
CREATE INDEX CONCURRENTLY idx_email_dashboard_metrics_type_time 
    ON email_dashboard_metrics(metric_type, time_period DESC);

-- Create refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_dashboard_metrics() RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY email_dashboard_metrics;
END;
$$ LANGUAGE plpgsql;

-- Database maintenance procedures

-- Procedure to partition email_logs by month
CREATE OR REPLACE FUNCTION create_email_logs_partition(
    partition_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    -- Calculate partition bounds
    start_date := DATE_TRUNC('month', partition_date);
    end_date := start_date + INTERVAL '1 month';
    
    -- Generate partition name
    partition_name := 'email_logs_y' || EXTRACT(YEAR FROM start_date) || 
                     'm' || LPAD(EXTRACT(MONTH FROM start_date)::TEXT, 2, '0');
    
    -- Create partition
    EXECUTE format('
        CREATE TABLE %I PARTITION OF email_logs
        FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
    
    -- Create indexes on new partition
    EXECUTE format('
        CREATE INDEX CONCURRENTLY %I ON %I(job_id)',
        'idx_' || partition_name || '_job_id', partition_name
    );
    
    EXECUTE format('
        CREATE INDEX CONCURRENTLY %I ON %I(recipient, status)',
        'idx_' || partition_name || '_recipient_status', partition_name
    );
    
    EXECUTE format('
        CREATE INDEX CONCURRENTLY %I ON %I(template_id)',
        'idx_' || partition_name || '_template_id', partition_name
    );
    
    EXECUTE format('
        CREATE INDEX CONCURRENTLY %I ON %I(status, created_at DESC)',
        'idx_' || partition_name || '_status_created', partition_name
    );
    
END;
$$ LANGUAGE plpgsql;

-- Procedure to drop old partitions
CREATE OR REPLACE FUNCTION cleanup_old_email_partitions(
    retention_months INTEGER DEFAULT 12
) RETURNS INTEGER AS $$
DECLARE
    partition_record RECORD;
    dropped_count INTEGER := 0;
    cutoff_date DATE;
BEGIN
    cutoff_date := DATE_TRUNC('month', CURRENT_DATE - (retention_months || ' months')::INTERVAL);
    
    -- Find and drop old partitions
    FOR partition_record IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'email_logs_y%m%'
        AND schemaname = 'public'
    LOOP
        -- Extract date from partition name and check if it's old enough
        DECLARE
            partition_date DATE;
            year_part INTEGER;
            month_part INTEGER;
        BEGIN
            year_part := SUBSTRING(partition_record.tablename FROM 'y(\d{4})m')::INTEGER;
            month_part := SUBSTRING(partition_record.tablename FROM 'm(\d{2})')::INTEGER;
            partition_date := make_date(year_part, month_part, 1);
            
            IF partition_date < cutoff_date THEN
                EXECUTE format('DROP TABLE %I', partition_record.tablename);
                dropped_count := dropped_count + 1;
            END IF;
        EXCEPTION
            WHEN OTHERS THEN
                -- Skip malformed partition names
                CONTINUE;
        END;
    END LOOP;
    
    RETURN dropped_count;
END;
$$ LANGUAGE plpgsql;

-- Create automatic maintenance job scheduling
-- This would typically be set up with pg_cron extension

-- Schedule daily dashboard metrics refresh
-- SELECT cron.schedule('refresh-dashboard-metrics', '0 */6 * * *', 'SELECT refresh_dashboard_metrics();');

-- Schedule monthly partition creation
-- SELECT cron.schedule('create-monthly-partition', '0 0 1 * *', 
--     'SELECT create_email_logs_partition(CURRENT_DATE + INTERVAL ''1 month'');');

-- Schedule quarterly partition cleanup
-- SELECT cron.schedule('cleanup-old-partitions', '0 2 1 */3 *', 
--     'SELECT cleanup_old_email_partitions(12);');

-- Performance monitoring queries

-- Query to identify slow templates
CREATE OR REPLACE VIEW slow_templates AS
SELECT 
    t.template_id,
    t.name,
    COUNT(el.id) as total_sends,
    AVG(el.processing_time) as avg_processing_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY el.processing_time) as p95_processing_time,
    MAX(el.processing_time) as max_processing_time
FROM email_templates t
JOIN email_logs el ON t.template_id = el.template_id
WHERE 
    el.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
    AND el.processing_time IS NOT NULL
GROUP BY t.template_id, t.name
HAVING AVG(el.processing_time) > 2.0  -- Templates taking more than 2 seconds on average
ORDER BY avg_processing_time DESC;

-- Query to identify high-bounce domains
CREATE OR REPLACE VIEW high_bounce_domains AS
SELECT 
    SPLIT_PART(recipient, '@', 2) as domain,
    COUNT(*) as total_emails,
    COUNT(CASE WHEN status = 'failed' AND error_message ILIKE '%bounce%' THEN 1 END) as bounces,
    (COUNT(CASE WHEN status = 'failed' AND error_message ILIKE '%bounce%' THEN 1 END)::FLOAT / COUNT(*)) * 100 as bounce_rate
FROM email_logs
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY SPLIT_PART(recipient, '@', 2)
HAVING COUNT(*) > 100  -- Minimum volume
AND (COUNT(CASE WHEN status = 'failed' AND error_message ILIKE '%bounce%' THEN 1 END)::FLOAT / COUNT(*)) > 0.05  -- > 5% bounce rate
ORDER BY bounce_rate DESC;
```
{% endraw %}

## Conclusion

Email API performance optimization requires comprehensive strategies spanning caching, database design, queue management, and infrastructure scaling. Organizations implementing these advanced optimization techniques consistently achieve 70-85% improvements in processing speed, 60-75% reduction in infrastructure costs, and 90-95% reliability improvements while maintaining exceptional user experiences across high-volume email operations.

Key success factors for email API optimization include:

1. **Multi-Tier Caching** - Intelligent cache management across memory, Redis, and database layers
2. **Database Optimization** - Partitioned tables, optimized indexes, and materialized views for high-performance queries
3. **Asynchronous Processing** - Priority queues and worker pools for efficient job processing
4. **Performance Monitoring** - Real-time metrics and automated optimization systems
5. **Scalable Architecture** - Horizontally scalable components with load balancing and redundancy

Organizations implementing these optimization strategies typically handle millions of emails daily while maintaining sub-second response times and 99.9% uptime. The investment in performance optimization pays dividends through reduced infrastructure costs, improved user experience, and the ability to scale operations efficiently.

The future of email API performance lies in AI-powered optimization systems that automatically tune caching strategies, database queries, and resource allocation based on real-time performance data. By implementing the frameworks outlined in this guide, engineering teams can build email infrastructure that consistently delivers exceptional performance at scale.

Remember that performance optimization is an ongoing process requiring continuous monitoring and refinement. Combining these technical optimizations with [professional email verification services](/services/) ensures both high performance and data quality, creating email systems that deliver exceptional results across all operational scenarios while maintaining optimal efficiency and reliability.