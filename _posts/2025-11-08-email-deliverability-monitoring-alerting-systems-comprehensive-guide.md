---
layout: post
title: "Email Deliverability Monitoring and Alerting Systems: Comprehensive Implementation Guide for High-Performance Email Infrastructure"
date: 2025-11-08 08:00:00 -0500
categories: email-deliverability monitoring automation infrastructure alerting
excerpt: "Build sophisticated email deliverability monitoring systems that proactively detect issues, trigger automated responses, and maintain optimal inbox placement rates. Learn to implement comprehensive alerting frameworks that protect your email program's performance and reputation."
---

# Email Deliverability Monitoring and Alerting Systems: Comprehensive Implementation Guide for High-Performance Email Infrastructure

Email deliverability monitoring has evolved from simple bounce tracking to comprehensive reputation management systems that provide real-time visibility into every aspect of email program performance. Modern marketing operations require sophisticated monitoring frameworks that detect issues before they impact business metrics, automatically trigger corrective actions, and provide actionable insights for continuous optimization.

Professional deliverability monitoring systems integrate data from multiple sources—mailbox providers, reputation services, authentication systems, and engagement platforms—to create holistic views of email program health. These systems enable proactive management of sender reputation, rapid response to delivery issues, and data-driven optimization of email infrastructure.

This comprehensive guide explores advanced monitoring and alerting strategies, covering real-time reputation tracking, automated incident response, predictive issue detection, and scalable infrastructure management that ensures consistent inbox placement across all major email providers.

## Deliverability Monitoring Architecture

### Core Monitoring Components

Effective deliverability monitoring requires integrated systems that track multiple performance indicators across the entire email lifecycle:

**Reputation Monitoring Framework:**
- Real-time tracking of IP and domain reputation across major mailbox providers
- Authentication status monitoring (SPF, DKIM, DMARC) with failure alerting
- Blacklist monitoring across commercial and community reputation databases
- Complaint rate tracking with provider-specific thresholds and escalation rules

**Performance Tracking Infrastructure:**
- Delivery rate monitoring with provider-specific benchmarking
- Bounce categorization and trend analysis with predictive failure detection
- Engagement rate tracking with statistical significance testing
- Infrastructure health monitoring including SMTP performance and queue status

### Comprehensive Monitoring System Implementation

Build sophisticated monitoring infrastructure that provides real-time visibility into deliverability performance:

{% raw %}
```python
# Advanced email deliverability monitoring and alerting system
import asyncio
import aiohttp
import logging
import json
import datetime
import statistics
import smtplib
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import sqlite3
import redis
import dns.resolver
import ssl
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict, deque

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class MonitoringMetric(Enum):
    DELIVERY_RATE = "delivery_rate"
    BOUNCE_RATE = "bounce_rate"
    COMPLAINT_RATE = "complaint_rate"
    REPUTATION_SCORE = "reputation_score"
    BLACKLIST_STATUS = "blacklist_status"
    AUTHENTICATION_FAILURE = "authentication_failure"
    ENGAGEMENT_RATE = "engagement_rate"
    INFRASTRUCTURE_HEALTH = "infrastructure_health"

class ProviderType(Enum):
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    YAHOO = "yahoo"
    APPLE = "apple"
    CORPORATE = "corporate"
    OTHER = "other"

@dataclass
class MonitoringThreshold:
    metric: MonitoringMetric
    warning_threshold: float
    critical_threshold: float
    duration_minutes: int = 5
    provider: Optional[ProviderType] = None
    comparison_operator: str = "less_than"  # less_than, greater_than, equals

@dataclass
class AlertRule:
    rule_id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 15
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class MonitoringAlert:
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime.datetime
    metrics: Dict[str, Any]
    affected_ips: List[str] = field(default_factory=list)
    affected_domains: List[str] = field(default_factory=list)
    resolution_status: str = "open"
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime.datetime] = None

@dataclass
class ReputationScore:
    ip_address: str
    domain: str
    provider: ProviderType
    score: float
    timestamp: datetime.datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeliverabilityMetrics:
    timestamp: datetime.datetime
    provider: ProviderType
    sent_count: int
    delivered_count: int
    bounced_count: int
    complaint_count: int
    engagement_count: int
    delivery_rate: float
    bounce_rate: float
    complaint_rate: float
    engagement_rate: float

class DeliverabilityMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_conn = sqlite3.connect('deliverability_monitoring.db', check_same_thread=False)
        self.redis_client = redis.Redis.from_url(config.get('redis_url', 'redis://localhost:6379'))
        
        # Initialize monitoring infrastructure
        self.initialize_database()
        
        # Monitoring configuration
        self.monitoring_intervals = {
            'reputation': config.get('reputation_check_interval', 300),  # 5 minutes
            'metrics': config.get('metrics_check_interval', 60),         # 1 minute
            'blacklist': config.get('blacklist_check_interval', 900),    # 15 minutes
            'authentication': config.get('auth_check_interval', 180)     # 3 minutes
        }
        
        # Alert tracking
        self.active_alerts = {}
        self.alert_cooldowns = {}
        self.notification_queue = asyncio.Queue()
        
        # Performance tracking
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minute-level data
        self.reputation_cache = {}
        
        # External service clients
        self.reputation_apis = {
            'senderscore': config.get('senderscore_api', {}),
            'barracuda': config.get('barracuda_api', {}),
            'microsoft_snds': config.get('snds_api', {})
        }
        
        self.blacklist_apis = [
            'zen.spamhaus.org',
            'bl.spamcop.net', 
            'b.barracudacentral.org',
            'dnsbl.sorbs.net'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring tasks
        asyncio.create_task(self.reputation_monitor())
        asyncio.create_task(self.metrics_monitor())
        asyncio.create_task(self.blacklist_monitor())
        asyncio.create_task(self.authentication_monitor())
        asyncio.create_task(self.alert_processor())
    
    def initialize_database(self):
        """Initialize database schema for monitoring data"""
        cursor = self.db_conn.cursor()
        
        # Monitoring thresholds table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_thresholds (
                threshold_id TEXT PRIMARY KEY,
                metric TEXT NOT NULL,
                warning_threshold REAL NOT NULL,
                critical_threshold REAL NOT NULL,
                duration_minutes INTEGER DEFAULT 5,
                provider TEXT,
                comparison_operator TEXT DEFAULT 'less_than',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alert rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                conditions TEXT NOT NULL,
                severity TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                cooldown_minutes INTEGER DEFAULT 15,
                escalation_rules TEXT DEFAULT '[]',
                notification_channels TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_alerts (
                alert_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                timestamp DATETIME NOT NULL,
                metrics TEXT DEFAULT '{}',
                affected_ips TEXT DEFAULT '[]',
                affected_domains TEXT DEFAULT '[]',
                resolution_status TEXT DEFAULT 'open',
                acknowledged_by TEXT,
                resolved_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (rule_id) REFERENCES alert_rules (rule_id)
            )
        ''')
        
        # Reputation scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reputation_scores (
                score_id TEXT PRIMARY KEY,
                ip_address TEXT NOT NULL,
                domain TEXT NOT NULL,
                provider TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Deliverability metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deliverability_metrics (
                metric_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                provider TEXT NOT NULL,
                sent_count INTEGER NOT NULL,
                delivered_count INTEGER NOT NULL,
                bounced_count INTEGER NOT NULL,
                complaint_count INTEGER NOT NULL,
                engagement_count INTEGER DEFAULT 0,
                delivery_rate REAL NOT NULL,
                bounce_rate REAL NOT NULL,
                complaint_rate REAL NOT NULL,
                engagement_rate REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Blacklist status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blacklist_status (
                status_id TEXT PRIMARY KEY,
                ip_address TEXT NOT NULL,
                blacklist_name TEXT NOT NULL,
                is_listed BOOLEAN NOT NULL,
                first_detected DATETIME,
                last_checked DATETIME NOT NULL,
                delisting_url TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Authentication status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authentication_status (
                auth_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                spf_status TEXT NOT NULL,
                dkim_status TEXT NOT NULL,
                dmarc_status TEXT NOT NULL,
                spf_record TEXT,
                dkim_record TEXT,
                dmarc_record TEXT,
                last_checked DATETIME NOT NULL,
                issues TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON monitoring_alerts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reputation_ip ON reputation_scores(ip_address)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON deliverability_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_blacklist_ip ON blacklist_status(ip_address)')
        
        self.db_conn.commit()
    
    async def add_monitoring_threshold(self, threshold: MonitoringThreshold) -> str:
        """Add a new monitoring threshold"""
        threshold_id = str(uuid.uuid4())
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            INSERT INTO monitoring_thresholds 
            (threshold_id, metric, warning_threshold, critical_threshold, duration_minutes, 
             provider, comparison_operator)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            threshold_id,
            threshold.metric.value,
            threshold.warning_threshold,
            threshold.critical_threshold,
            threshold.duration_minutes,
            threshold.provider.value if threshold.provider else None,
            threshold.comparison_operator
        ))
        
        self.db_conn.commit()
        self.logger.info(f"Added monitoring threshold: {threshold.metric.value}")
        
        return threshold_id
    
    async def add_alert_rule(self, rule: AlertRule) -> str:
        """Add a new alert rule"""
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_rules 
            (rule_id, name, description, conditions, severity, enabled, 
             cooldown_minutes, escalation_rules, notification_channels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rule.rule_id,
            rule.name,
            rule.description,
            json.dumps(rule.conditions),
            rule.severity.value,
            rule.enabled,
            rule.cooldown_minutes,
            json.dumps(rule.escalation_rules),
            json.dumps(rule.notification_channels)
        ))
        
        self.db_conn.commit()
        self.logger.info(f"Added alert rule: {rule.name}")
        
        return rule.rule_id
    
    async def reputation_monitor(self):
        """Monitor sender reputation across various sources"""
        while True:
            try:
                # Get configured IP addresses and domains to monitor
                monitored_ips = self.config.get('monitored_ips', [])
                monitored_domains = self.config.get('monitored_domains', [])
                
                tasks = []
                
                # Monitor IP reputation
                for ip in monitored_ips:
                    for provider in ProviderType:
                        tasks.append(self.check_ip_reputation(ip, provider))
                
                # Monitor domain reputation
                for domain in monitored_domains:
                    for provider in ProviderType:
                        tasks.append(self.check_domain_reputation(domain, provider))
                
                if tasks:
                    reputation_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and trigger alerts if needed
                    for result in reputation_results:
                        if isinstance(result, ReputationScore):
                            await self.process_reputation_score(result)
                        elif isinstance(result, Exception):
                            self.logger.error(f"Reputation check error: {str(result)}")
                
                # Wait until next check
                await asyncio.sleep(self.monitoring_intervals['reputation'])
                
            except Exception as e:
                self.logger.error(f"Error in reputation monitoring: {str(e)}")
                await asyncio.sleep(60)  # Short retry delay
    
    async def check_ip_reputation(self, ip_address: str, provider: ProviderType) -> Optional[ReputationScore]:
        """Check IP reputation with specific provider"""
        try:
            reputation_score = None
            source = "unknown"
            metadata = {}
            
            if provider == ProviderType.GMAIL:
                # Check Gmail Postmaster Tools API (requires setup)
                reputation_score = await self.check_gmail_reputation(ip_address)
                source = "gmail_postmaster"
                
            elif provider == ProviderType.OUTLOOK:
                # Check Microsoft SNDS
                reputation_score = await self.check_microsoft_snds(ip_address)
                source = "microsoft_snds"
                
            elif provider == ProviderType.YAHOO:
                # Yahoo doesn't provide direct API, use general reputation services
                reputation_score = await self.check_senderscore_reputation(ip_address)
                source = "senderscore"
                
            else:
                # Use general reputation services
                reputation_score = await self.check_general_reputation(ip_address)
                source = "general"
            
            if reputation_score is not None:
                score_record = ReputationScore(
                    ip_address=ip_address,
                    domain="",
                    provider=provider,
                    score=reputation_score,
                    timestamp=datetime.datetime.utcnow(),
                    source=source,
                    metadata=metadata
                )
                
                # Store in database
                await self.store_reputation_score(score_record)
                
                return score_record
                
        except Exception as e:
            self.logger.error(f"Error checking IP reputation {ip_address} for {provider.value}: {str(e)}")
        
        return None
    
    async def check_gmail_reputation(self, ip_address: str) -> Optional[float]:
        """Check reputation via Gmail Postmaster Tools API"""
        # This would integrate with Gmail Postmaster Tools API
        # For demo purposes, we'll simulate
        try:
            # Simulate API call delay
            await asyncio.sleep(0.1)
            
            # Return simulated reputation score (0.0 to 1.0)
            # In real implementation, this would call the actual API
            return 0.85  # Good reputation
            
        except Exception as e:
            self.logger.error(f"Gmail reputation check failed for {ip_address}: {str(e)}")
            return None
    
    async def check_microsoft_snds(self, ip_address: str) -> Optional[float]:
        """Check reputation via Microsoft SNDS"""
        try:
            snds_config = self.reputation_apis.get('microsoft_snds', {})
            if not snds_config:
                return None
            
            # Simulate SNDS API call
            await asyncio.sleep(0.1)
            
            # Return simulated score
            return 0.78
            
        except Exception as e:
            self.logger.error(f"Microsoft SNDS check failed for {ip_address}: {str(e)}")
            return None
    
    async def check_senderscore_reputation(self, ip_address: str) -> Optional[float]:
        """Check reputation via SenderScore/Validity"""
        try:
            # Simulate SenderScore API call
            await asyncio.sleep(0.1)
            
            # Convert typical SenderScore (0-100) to 0.0-1.0 scale
            senderscore = 82  # Simulated score
            return senderscore / 100.0
            
        except Exception as e:
            self.logger.error(f"SenderScore check failed for {ip_address}: {str(e)}")
            return None
    
    async def check_general_reputation(self, ip_address: str) -> Optional[float]:
        """Check general IP reputation from multiple sources"""
        try:
            # This would aggregate multiple reputation sources
            # For demo, return a simulated aggregate score
            await asyncio.sleep(0.1)
            return 0.72
            
        except Exception as e:
            self.logger.error(f"General reputation check failed for {ip_address}: {str(e)}")
            return None
    
    async def check_domain_reputation(self, domain: str, provider: ProviderType) -> Optional[ReputationScore]:
        """Check domain reputation with specific provider"""
        try:
            # Domain reputation checking logic
            reputation_score = await self.check_domain_reputation_score(domain, provider)
            
            if reputation_score is not None:
                score_record = ReputationScore(
                    ip_address="",
                    domain=domain,
                    provider=provider,
                    score=reputation_score,
                    timestamp=datetime.datetime.utcnow(),
                    source=f"{provider.value}_domain_reputation"
                )
                
                await self.store_reputation_score(score_record)
                return score_record
                
        except Exception as e:
            self.logger.error(f"Error checking domain reputation {domain} for {provider.value}: {str(e)}")
        
        return None
    
    async def check_domain_reputation_score(self, domain: str, provider: ProviderType) -> Optional[float]:
        """Get domain reputation score from provider"""
        # Simulate domain reputation check
        await asyncio.sleep(0.1)
        return 0.88  # Good domain reputation
    
    async def store_reputation_score(self, score: ReputationScore):
        """Store reputation score in database"""
        cursor = self.db_conn.cursor()
        score_id = str(uuid.uuid4())
        
        cursor.execute('''
            INSERT INTO reputation_scores 
            (score_id, ip_address, domain, provider, score, timestamp, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            score_id,
            score.ip_address,
            score.domain,
            score.provider.value,
            score.score,
            score.timestamp,
            score.source,
            json.dumps(score.metadata)
        ))
        
        self.db_conn.commit()
    
    async def process_reputation_score(self, score: ReputationScore):
        """Process reputation score and check against thresholds"""
        # Cache the score for quick access
        cache_key = f"reputation:{score.ip_address or score.domain}:{score.provider.value}"
        self.reputation_cache[cache_key] = score
        
        # Check against configured thresholds
        await self.check_reputation_thresholds(score)
    
    async def check_reputation_thresholds(self, score: ReputationScore):
        """Check reputation score against configured thresholds"""
        cursor = self.db_conn.cursor()
        
        # Find matching thresholds
        cursor.execute('''
            SELECT threshold_id, warning_threshold, critical_threshold, comparison_operator
            FROM monitoring_thresholds 
            WHERE metric = ? AND (provider IS NULL OR provider = ?)
        ''', (MonitoringMetric.REPUTATION_SCORE.value, score.provider.value))
        
        thresholds = cursor.fetchall()
        
        for threshold_id, warning_threshold, critical_threshold, comparison_operator in thresholds:
            trigger_level = None
            
            if comparison_operator == "less_than":
                if score.score < critical_threshold:
                    trigger_level = AlertSeverity.CRITICAL
                elif score.score < warning_threshold:
                    trigger_level = AlertSeverity.WARNING
                    
            elif comparison_operator == "greater_than":
                if score.score > critical_threshold:
                    trigger_level = AlertSeverity.CRITICAL
                elif score.score > warning_threshold:
                    trigger_level = AlertSeverity.WARNING
            
            if trigger_level:
                await self.trigger_reputation_alert(score, trigger_level, threshold_id)
    
    async def trigger_reputation_alert(self, score: ReputationScore, severity: AlertSeverity, threshold_id: str):
        """Trigger alert for reputation threshold breach"""
        alert_id = str(uuid.uuid4())
        
        asset = score.ip_address or score.domain
        asset_type = "IP" if score.ip_address else "Domain"
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            rule_id=f"reputation_threshold_{threshold_id}",
            severity=severity,
            title=f"{asset_type} Reputation Alert: {asset}",
            description=f"{asset_type} {asset} reputation score {score.score:.2f} below threshold for {score.provider.value}",
            timestamp=datetime.datetime.utcnow(),
            metrics={
                'reputation_score': score.score,
                'provider': score.provider.value,
                'source': score.source,
                'asset': asset,
                'asset_type': asset_type.lower()
            },
            affected_ips=[score.ip_address] if score.ip_address else [],
            affected_domains=[score.domain] if score.domain else []
        )
        
        await self.process_alert(alert)
    
    async def metrics_monitor(self):
        """Monitor delivery metrics and performance indicators"""
        while True:
            try:
                # Collect metrics from various sources
                metrics_data = await self.collect_delivery_metrics()
                
                # Process each provider's metrics
                for metrics in metrics_data:
                    await self.process_delivery_metrics(metrics)
                    await self.check_delivery_thresholds(metrics)
                
                # Wait until next check
                await asyncio.sleep(self.monitoring_intervals['metrics'])
                
            except Exception as e:
                self.logger.error(f"Error in metrics monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def collect_delivery_metrics(self) -> List[DeliverabilityMetrics]:
        """Collect delivery metrics from email service providers"""
        current_time = datetime.datetime.utcnow()
        
        # This would typically integrate with your ESP's API
        # For demonstration, we'll simulate metrics
        simulated_metrics = [
            DeliverabilityMetrics(
                timestamp=current_time,
                provider=ProviderType.GMAIL,
                sent_count=10000,
                delivered_count=9850,
                bounced_count=100,
                complaint_count=5,
                engagement_count=2455,
                delivery_rate=0.985,
                bounce_rate=0.010,
                complaint_rate=0.0005,
                engagement_rate=0.249
            ),
            DeliverabilityMetrics(
                timestamp=current_time,
                provider=ProviderType.OUTLOOK,
                sent_count=8500,
                delivered_count=8200,
                bounced_count=250,
                complaint_count=8,
                engagement_count=1804,
                delivery_rate=0.965,
                bounce_rate=0.029,
                complaint_rate=0.0009,
                engagement_rate=0.220
            ),
            DeliverabilityMetrics(
                timestamp=current_time,
                provider=ProviderType.YAHOO,
                sent_count=6200,
                delivered_count=5900,
                bounced_count=280,
                complaint_count=12,
                engagement_count=1180,
                delivery_rate=0.952,
                bounce_rate=0.045,
                complaint_rate=0.0019,
                engagement_rate=0.200
            )
        ]
        
        return simulated_metrics
    
    async def process_delivery_metrics(self, metrics: DeliverabilityMetrics):
        """Process and store delivery metrics"""
        # Store in database
        metric_id = str(uuid.uuid4())
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            INSERT INTO deliverability_metrics 
            (metric_id, timestamp, provider, sent_count, delivered_count, bounced_count,
             complaint_count, engagement_count, delivery_rate, bounce_rate, complaint_rate, engagement_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric_id,
            metrics.timestamp,
            metrics.provider.value,
            metrics.sent_count,
            metrics.delivered_count,
            metrics.bounced_count,
            metrics.complaint_count,
            metrics.engagement_count,
            metrics.delivery_rate,
            metrics.bounce_rate,
            metrics.complaint_rate,
            metrics.engagement_rate
        ))
        
        self.db_conn.commit()
        
        # Update in-memory buffer for real-time analysis
        buffer_key = f"metrics:{metrics.provider.value}"
        self.metrics_buffer[buffer_key].append({
            'timestamp': metrics.timestamp,
            'delivery_rate': metrics.delivery_rate,
            'bounce_rate': metrics.bounce_rate,
            'complaint_rate': metrics.complaint_rate,
            'engagement_rate': metrics.engagement_rate
        })
    
    async def check_delivery_thresholds(self, metrics: DeliverabilityMetrics):
        """Check delivery metrics against configured thresholds"""
        cursor = self.db_conn.cursor()
        
        # Check delivery rate threshold
        cursor.execute('''
            SELECT threshold_id, warning_threshold, critical_threshold, comparison_operator, duration_minutes
            FROM monitoring_thresholds 
            WHERE metric = ? AND (provider IS NULL OR provider = ?)
        ''', (MonitoringMetric.DELIVERY_RATE.value, metrics.provider.value))
        
        thresholds = cursor.fetchall()
        
        for threshold_id, warning_threshold, critical_threshold, comparison_operator, duration_minutes in thresholds:
            # Check if threshold is breached for sustained period
            if await self.is_threshold_breached(
                metrics.provider, 
                'delivery_rate', 
                metrics.delivery_rate, 
                warning_threshold, 
                critical_threshold,
                comparison_operator,
                duration_minutes
            ):
                severity = AlertSeverity.CRITICAL if metrics.delivery_rate < critical_threshold else AlertSeverity.WARNING
                await self.trigger_delivery_alert(metrics, severity, 'delivery_rate', threshold_id)
        
        # Check bounce rate threshold
        cursor.execute('''
            SELECT threshold_id, warning_threshold, critical_threshold, comparison_operator, duration_minutes
            FROM monitoring_thresholds 
            WHERE metric = ? AND (provider IS NULL OR provider = ?)
        ''', (MonitoringMetric.BOUNCE_RATE.value, metrics.provider.value))
        
        thresholds = cursor.fetchall()
        
        for threshold_id, warning_threshold, critical_threshold, comparison_operator, duration_minutes in thresholds:
            if await self.is_threshold_breached(
                metrics.provider, 
                'bounce_rate', 
                metrics.bounce_rate, 
                warning_threshold, 
                critical_threshold,
                comparison_operator,
                duration_minutes
            ):
                severity = AlertSeverity.CRITICAL if metrics.bounce_rate > critical_threshold else AlertSeverity.WARNING
                await self.trigger_delivery_alert(metrics, severity, 'bounce_rate', threshold_id)
        
        # Check complaint rate threshold
        cursor.execute('''
            SELECT threshold_id, warning_threshold, critical_threshold, comparison_operator, duration_minutes
            FROM monitoring_thresholds 
            WHERE metric = ? AND (provider IS NULL OR provider = ?)
        ''', (MonitoringMetric.COMPLAINT_RATE.value, metrics.provider.value))
        
        thresholds = cursor.fetchall()
        
        for threshold_id, warning_threshold, critical_threshold, comparison_operator, duration_minutes in thresholds:
            if await self.is_threshold_breached(
                metrics.provider, 
                'complaint_rate', 
                metrics.complaint_rate, 
                warning_threshold, 
                critical_threshold,
                comparison_operator,
                duration_minutes
            ):
                severity = AlertSeverity.CRITICAL if metrics.complaint_rate > critical_threshold else AlertSeverity.WARNING
                await self.trigger_delivery_alert(metrics, severity, 'complaint_rate', threshold_id)
    
    async def is_threshold_breached(self, provider: ProviderType, metric_name: str, 
                                  current_value: float, warning_threshold: float,
                                  critical_threshold: float, comparison_operator: str,
                                  duration_minutes: int) -> bool:
        """Check if threshold has been breached for sustained duration"""
        buffer_key = f"metrics:{provider.value}"
        
        if buffer_key not in self.metrics_buffer:
            return False
        
        recent_metrics = list(self.metrics_buffer[buffer_key])
        
        if len(recent_metrics) < duration_minutes:
            return False  # Not enough data points
        
        # Check last N minutes of data
        cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=duration_minutes)
        
        relevant_metrics = [
            m for m in recent_metrics 
            if m['timestamp'] >= cutoff_time and metric_name in m
        ]
        
        if len(relevant_metrics) < duration_minutes * 0.8:  # Allow for some missing data
            return False
        
        # Check if threshold is consistently breached
        breach_count = 0
        for metric_data in relevant_metrics:
            value = metric_data[metric_name]
            
            if comparison_operator == "less_than":
                if value < warning_threshold:
                    breach_count += 1
            elif comparison_operator == "greater_than":
                if value > warning_threshold:
                    breach_count += 1
        
        # Consider threshold breached if 80% of recent values breach threshold
        return (breach_count / len(relevant_metrics)) >= 0.8
    
    async def trigger_delivery_alert(self, metrics: DeliverabilityMetrics, 
                                   severity: AlertSeverity, metric_type: str,
                                   threshold_id: str):
        """Trigger alert for delivery metric threshold breach"""
        alert_id = str(uuid.uuid4())
        
        metric_value = getattr(metrics, metric_type)
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            rule_id=f"delivery_threshold_{threshold_id}",
            severity=severity,
            title=f"Delivery Alert: {metric_type.replace('_', ' ').title()} for {metrics.provider.value}",
            description=f"{metric_type.replace('_', ' ').title()} for {metrics.provider.value} is {metric_value:.3f}",
            timestamp=datetime.datetime.utcnow(),
            metrics={
                'provider': metrics.provider.value,
                'metric_type': metric_type,
                'metric_value': metric_value,
                'delivery_rate': metrics.delivery_rate,
                'bounce_rate': metrics.bounce_rate,
                'complaint_rate': metrics.complaint_rate,
                'sent_count': metrics.sent_count
            }
        )
        
        await self.process_alert(alert)
    
    async def blacklist_monitor(self):
        """Monitor IP addresses against major blacklists"""
        while True:
            try:
                monitored_ips = self.config.get('monitored_ips', [])
                
                for ip in monitored_ips:
                    for blacklist in self.blacklist_apis:
                        try:
                            is_listed = await self.check_blacklist_status(ip, blacklist)
                            await self.store_blacklist_status(ip, blacklist, is_listed)
                            
                            if is_listed:
                                await self.trigger_blacklist_alert(ip, blacklist)
                                
                        except Exception as e:
                            self.logger.error(f"Blacklist check failed for {ip} on {blacklist}: {str(e)}")
                
                await asyncio.sleep(self.monitoring_intervals['blacklist'])
                
            except Exception as e:
                self.logger.error(f"Error in blacklist monitoring: {str(e)}")
                await asyncio.sleep(300)  # 5 minute retry delay
    
    async def check_blacklist_status(self, ip_address: str, blacklist: str) -> bool:
        """Check if IP is listed on specific blacklist"""
        try:
            # Reverse IP for DNS lookup
            reversed_ip = '.'.join(ip_address.split('.')[::-1])
            query_domain = f"{reversed_ip}.{blacklist}"
            
            # Perform DNS lookup
            resolver = dns.resolver.Resolver()
            resolver.timeout = 10
            
            try:
                answers = resolver.resolve(query_domain, 'A')
                # If we get an answer, IP is listed
                return True
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                # IP is not listed
                return False
                
        except Exception as e:
            self.logger.error(f"Blacklist DNS check failed for {ip_address} on {blacklist}: {str(e)}")
            return False
    
    async def store_blacklist_status(self, ip_address: str, blacklist: str, is_listed: bool):
        """Store blacklist status in database"""
        cursor = self.db_conn.cursor()
        
        # Check if we have existing record
        cursor.execute('''
            SELECT status_id, is_listed, first_detected FROM blacklist_status 
            WHERE ip_address = ? AND blacklist_name = ?
        ''', (ip_address, blacklist))
        
        existing = cursor.fetchone()
        current_time = datetime.datetime.utcnow()
        
        if existing:
            status_id, previous_listed, first_detected = existing
            
            if is_listed and not previous_listed:
                # Newly listed
                first_detected = current_time
            elif not is_listed:
                # Not listed (or delisted)
                first_detected = None
            
            # Update existing record
            cursor.execute('''
                UPDATE blacklist_status 
                SET is_listed = ?, first_detected = ?, last_checked = ?
                WHERE status_id = ?
            ''', (is_listed, first_detected, current_time, status_id))
        else:
            # Create new record
            status_id = str(uuid.uuid4())
            first_detected = current_time if is_listed else None
            
            cursor.execute('''
                INSERT INTO blacklist_status 
                (status_id, ip_address, blacklist_name, is_listed, first_detected, last_checked)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (status_id, ip_address, blacklist, is_listed, first_detected, current_time))
        
        self.db_conn.commit()
    
    async def trigger_blacklist_alert(self, ip_address: str, blacklist: str):
        """Trigger alert for blacklist detection"""
        alert_id = str(uuid.uuid4())
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            rule_id="blacklist_detection",
            severity=AlertSeverity.CRITICAL,
            title=f"Blacklist Detection: {ip_address}",
            description=f"IP address {ip_address} has been detected on blacklist {blacklist}",
            timestamp=datetime.datetime.utcnow(),
            metrics={
                'ip_address': ip_address,
                'blacklist': blacklist,
                'detection_time': datetime.datetime.utcnow().isoformat()
            },
            affected_ips=[ip_address]
        )
        
        await self.process_alert(alert)
    
    async def authentication_monitor(self):
        """Monitor email authentication (SPF, DKIM, DMARC) status"""
        while True:
            try:
                monitored_domains = self.config.get('monitored_domains', [])
                
                for domain in monitored_domains:
                    auth_status = await self.check_authentication_status(domain)
                    await self.store_authentication_status(domain, auth_status)
                    
                    if auth_status.get('issues'):
                        await self.trigger_authentication_alert(domain, auth_status)
                
                await asyncio.sleep(self.monitoring_intervals['authentication'])
                
            except Exception as e:
                self.logger.error(f"Error in authentication monitoring: {str(e)}")
                await asyncio.sleep(300)
    
    async def check_authentication_status(self, domain: str) -> Dict[str, Any]:
        """Check SPF, DKIM, and DMARC records for domain"""
        auth_status = {
            'spf_status': 'unknown',
            'dkim_status': 'unknown', 
            'dmarc_status': 'unknown',
            'spf_record': None,
            'dkim_record': None,
            'dmarc_record': None,
            'issues': []
        }
        
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = 10
            
            # Check SPF record
            try:
                spf_answers = resolver.resolve(domain, 'TXT')
                for rdata in spf_answers:
                    txt_record = str(rdata).strip('"')
                    if txt_record.startswith('v=spf1'):
                        auth_status['spf_record'] = txt_record
                        auth_status['spf_status'] = 'valid'
                        break
                else:
                    auth_status['spf_status'] = 'missing'
                    auth_status['issues'].append('SPF record not found')
                    
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                auth_status['spf_status'] = 'missing'
                auth_status['issues'].append('SPF record not found')
            
            # Check DMARC record
            try:
                dmarc_domain = f"_dmarc.{domain}"
                dmarc_answers = resolver.resolve(dmarc_domain, 'TXT')
                for rdata in dmarc_answers:
                    txt_record = str(rdata).strip('"')
                    if txt_record.startswith('v=DMARC1'):
                        auth_status['dmarc_record'] = txt_record
                        auth_status['dmarc_status'] = 'valid'
                        break
                else:
                    auth_status['dmarc_status'] = 'missing'
                    auth_status['issues'].append('DMARC record not found')
                    
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                auth_status['dmarc_status'] = 'missing'
                auth_status['issues'].append('DMARC record not found')
            
            # Check DKIM (requires knowledge of selector, using common ones)
            dkim_selectors = self.config.get('dkim_selectors', ['default', 'selector1', 'selector2', 'k1'])
            dkim_found = False
            
            for selector in dkim_selectors:
                try:
                    dkim_domain = f"{selector}._domainkey.{domain}"
                    dkim_answers = resolver.resolve(dkim_domain, 'TXT')
                    for rdata in dkim_answers:
                        txt_record = str(rdata).strip('"')
                        if 'k=' in txt_record or 'p=' in txt_record:
                            auth_status['dkim_record'] = txt_record
                            auth_status['dkim_status'] = 'valid'
                            dkim_found = True
                            break
                    if dkim_found:
                        break
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                    continue
            
            if not dkim_found:
                auth_status['dkim_status'] = 'missing'
                auth_status['issues'].append('DKIM record not found for common selectors')
                
        except Exception as e:
            self.logger.error(f"Authentication check failed for {domain}: {str(e)}")
            auth_status['issues'].append(f"DNS resolution error: {str(e)}")
        
        return auth_status
    
    async def store_authentication_status(self, domain: str, auth_status: Dict[str, Any]):
        """Store authentication status in database"""
        cursor = self.db_conn.cursor()
        auth_id = str(uuid.uuid4())
        
        cursor.execute('''
            INSERT INTO authentication_status 
            (auth_id, domain, spf_status, dkim_status, dmarc_status,
             spf_record, dkim_record, dmarc_record, last_checked, issues)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            auth_id,
            domain,
            auth_status['spf_status'],
            auth_status['dkim_status'],
            auth_status['dmarc_status'],
            auth_status['spf_record'],
            auth_status['dkim_record'],
            auth_status['dmarc_record'],
            datetime.datetime.utcnow(),
            json.dumps(auth_status['issues'])
        ))
        
        self.db_conn.commit()
    
    async def trigger_authentication_alert(self, domain: str, auth_status: Dict[str, Any]):
        """Trigger alert for authentication issues"""
        alert_id = str(uuid.uuid4())
        
        issues_text = ', '.join(auth_status['issues'])
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            rule_id="authentication_issues",
            severity=AlertSeverity.WARNING,
            title=f"Authentication Issues: {domain}",
            description=f"Domain {domain} has authentication issues: {issues_text}",
            timestamp=datetime.datetime.utcnow(),
            metrics={
                'domain': domain,
                'spf_status': auth_status['spf_status'],
                'dkim_status': auth_status['dkim_status'],
                'dmarc_status': auth_status['dmarc_status'],
                'issues': auth_status['issues']
            },
            affected_domains=[domain]
        )
        
        await self.process_alert(alert)
    
    async def process_alert(self, alert: MonitoringAlert):
        """Process and route alerts through notification system"""
        # Check alert cooldown
        cooldown_key = f"alert_cooldown:{alert.rule_id}"
        
        if cooldown_key in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[cooldown_key]
            if (datetime.datetime.utcnow() - last_alert_time).total_seconds() < 900:  # 15 minutes default
                return  # Skip alert due to cooldown
        
        # Store alert in database
        await self.store_alert(alert)
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        
        # Set cooldown
        self.alert_cooldowns[cooldown_key] = datetime.datetime.utcnow()
        
        # Queue for notification
        await self.notification_queue.put(alert)
        
        self.logger.warning(f"Alert triggered: {alert.title} ({alert.severity.value})")
    
    async def store_alert(self, alert: MonitoringAlert):
        """Store alert in database"""
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            INSERT INTO monitoring_alerts 
            (alert_id, rule_id, severity, title, description, timestamp, metrics,
             affected_ips, affected_domains, resolution_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.rule_id,
            alert.severity.value,
            alert.title,
            alert.description,
            alert.timestamp,
            json.dumps(alert.metrics),
            json.dumps(alert.affected_ips),
            json.dumps(alert.affected_domains),
            alert.resolution_status
        ))
        
        self.db_conn.commit()
    
    async def alert_processor(self):
        """Process alert notifications"""
        while True:
            try:
                alert = await self.notification_queue.get()
                
                # Send notifications through configured channels
                await self.send_alert_notifications(alert)
                
                # Check for escalation rules
                await self.process_alert_escalation(alert)
                
                self.notification_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing alert notifications: {str(e)}")
    
    async def send_alert_notifications(self, alert: MonitoringAlert):
        """Send alert notifications through configured channels"""
        notification_channels = self.config.get('notification_channels', {})
        
        # Email notifications
        if 'email' in notification_channels:
            await self.send_email_notification(alert, notification_channels['email'])
        
        # Slack notifications
        if 'slack' in notification_channels:
            await self.send_slack_notification(alert, notification_channels['slack'])
        
        # SMS notifications for critical alerts
        if alert.severity == AlertSeverity.CRITICAL and 'sms' in notification_channels:
            await self.send_sms_notification(alert, notification_channels['sms'])
        
        # Webhook notifications
        if 'webhook' in notification_channels:
            await self.send_webhook_notification(alert, notification_channels['webhook'])
    
    async def send_email_notification(self, alert: MonitoringAlert, email_config: Dict[str, Any]):
        """Send email notification for alert"""
        try:
            smtp_server = email_config.get('smtp_server', 'localhost')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username', '')
            password = email_config.get('password', '')
            recipients = email_config.get('recipients', [])
            
            if not recipients:
                return
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('sender', 'alerts@yourdomain.com')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
            Alert Details:
            
            Severity: {alert.severity.value.upper()}
            Title: {alert.title}
            Description: {alert.description}
            Timestamp: {alert.timestamp}
            
            Metrics:
            {json.dumps(alert.metrics, indent=2)}
            
            Affected IPs: {', '.join(alert.affected_ips) if alert.affected_ips else 'None'}
            Affected Domains: {', '.join(alert.affected_domains) if alert.affected_domains else 'None'}
            
            Alert ID: {alert.alert_id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
    
    async def send_slack_notification(self, alert: MonitoringAlert, slack_config: Dict[str, Any]):
        """Send Slack notification for alert"""
        try:
            webhook_url = slack_config.get('webhook_url')
            if not webhook_url:
                return
            
            # Determine emoji and color based on severity
            severity_config = {
                AlertSeverity.INFO: {'color': 'good', 'emoji': ':information_source:'},
                AlertSeverity.WARNING: {'color': 'warning', 'emoji': ':warning:'},
                AlertSeverity.ERROR: {'color': 'danger', 'emoji': ':exclamation:'},
                AlertSeverity.CRITICAL: {'color': 'danger', 'emoji': ':rotating_light:'}
            }
            
            config = severity_config.get(alert.severity, severity_config[AlertSeverity.INFO])
            
            # Create Slack message
            payload = {
                'attachments': [{
                    'color': config['color'],
                    'title': f"{config['emoji']} {alert.title}",
                    'text': alert.description,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Timestamp', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), 'short': True},
                        {'title': 'Alert ID', 'value': alert.alert_id, 'short': True}
                    ],
                    'footer': 'Deliverability Monitor',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            # Add affected assets if any
            if alert.affected_ips:
                payload['attachments'][0]['fields'].append({
                    'title': 'Affected IPs',
                    'value': ', '.join(alert.affected_ips),
                    'short': False
                })
            
            if alert.affected_domains:
                payload['attachments'][0]['fields'].append({
                    'title': 'Affected Domains',
                    'value': ', '.join(alert.affected_domains),
                    'short': False
                })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack notification sent for alert {alert.alert_id}")
                    else:
                        self.logger.error(f"Slack notification failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {str(e)}")
    
    async def send_webhook_notification(self, alert: MonitoringAlert, webhook_config: Dict[str, Any]):
        """Send webhook notification for alert"""
        try:
            webhook_url = webhook_config.get('url')
            if not webhook_url:
                return
            
            # Prepare webhook payload
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat(),
                'metrics': alert.metrics,
                'affected_ips': alert.affected_ips,
                'affected_domains': alert.affected_domains,
                'resolution_status': alert.resolution_status
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'DeliverabilityMonitor/1.0'
            }
            
            # Add authentication if configured
            if 'auth_header' in webhook_config:
                headers['Authorization'] = webhook_config['auth_header']
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                    else:
                        self.logger.error(f"Webhook notification failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {str(e)}")
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        try:
            current_time = datetime.datetime.utcnow()
            
            # Get recent delivery metrics
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT provider, AVG(delivery_rate), AVG(bounce_rate), AVG(complaint_rate), AVG(engagement_rate)
                FROM deliverability_metrics 
                WHERE timestamp >= ?
                GROUP BY provider
            ''', (current_time - datetime.timedelta(hours=1),))
            
            delivery_metrics = {}
            for row in cursor.fetchall():
                provider, delivery_rate, bounce_rate, complaint_rate, engagement_rate = row
                delivery_metrics[provider] = {
                    'delivery_rate': round(delivery_rate, 4),
                    'bounce_rate': round(bounce_rate, 4),
                    'complaint_rate': round(complaint_rate, 6),
                    'engagement_rate': round(engagement_rate, 4)
                }
            
            # Get reputation scores
            cursor.execute('''
                SELECT ip_address, domain, provider, score, timestamp
                FROM reputation_scores 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', (current_time - datetime.timedelta(hours=24),))
            
            reputation_scores = []
            for row in cursor.fetchall():
                ip_address, domain, provider, score, timestamp = row
                reputation_scores.append({
                    'asset': ip_address or domain,
                    'asset_type': 'ip' if ip_address else 'domain',
                    'provider': provider,
                    'score': score,
                    'timestamp': timestamp
                })
            
            # Get active alerts
            cursor.execute('''
                SELECT alert_id, severity, title, description, timestamp, resolution_status
                FROM monitoring_alerts 
                WHERE resolution_status = 'open'
                ORDER BY timestamp DESC
                LIMIT 20
            ''', )
            
            active_alerts = []
            for row in cursor.fetchall():
                alert_id, severity, title, description, timestamp, resolution_status = row
                active_alerts.append({
                    'alert_id': alert_id,
                    'severity': severity,
                    'title': title,
                    'description': description,
                    'timestamp': timestamp,
                    'resolution_status': resolution_status
                })
            
            # Get blacklist status
            cursor.execute('''
                SELECT ip_address, blacklist_name, is_listed, first_detected, last_checked
                FROM blacklist_status 
                WHERE is_listed = 1
                ORDER BY first_detected DESC
            ''')
            
            blacklist_issues = []
            for row in cursor.fetchall():
                ip_address, blacklist_name, is_listed, first_detected, last_checked = row
                blacklist_issues.append({
                    'ip_address': ip_address,
                    'blacklist': blacklist_name,
                    'first_detected': first_detected,
                    'last_checked': last_checked
                })
            
            return {
                'timestamp': current_time.isoformat(),
                'delivery_metrics': delivery_metrics,
                'reputation_scores': reputation_scores,
                'active_alerts': active_alerts,
                'blacklist_issues': blacklist_issues,
                'system_status': 'operational'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {str(e)}")
            return {'error': str(e)}

# Configuration and monitoring rule builders
class MonitoringRuleBuilder:
    """Builder class for creating monitoring rules and thresholds"""
    
    @staticmethod
    def create_default_thresholds() -> List[MonitoringThreshold]:
        """Create default monitoring thresholds"""
        return [
            # Delivery rate thresholds
            MonitoringThreshold(
                metric=MonitoringMetric.DELIVERY_RATE,
                warning_threshold=0.95,
                critical_threshold=0.90,
                duration_minutes=5,
                comparison_operator="less_than"
            ),
            
            # Bounce rate thresholds  
            MonitoringThreshold(
                metric=MonitoringMetric.BOUNCE_RATE,
                warning_threshold=0.05,
                critical_threshold=0.10,
                duration_minutes=5,
                comparison_operator="greater_than"
            ),
            
            # Complaint rate thresholds
            MonitoringThreshold(
                metric=MonitoringMetric.COMPLAINT_RATE,
                warning_threshold=0.001,  # 0.1%
                critical_threshold=0.003,  # 0.3%
                duration_minutes=10,
                comparison_operator="greater_than"
            ),
            
            # Reputation score thresholds
            MonitoringThreshold(
                metric=MonitoringMetric.REPUTATION_SCORE,
                warning_threshold=0.70,
                critical_threshold=0.50,
                duration_minutes=1,
                comparison_operator="less_than"
            ),
            
            # Provider-specific Gmail thresholds
            MonitoringThreshold(
                metric=MonitoringMetric.DELIVERY_RATE,
                warning_threshold=0.98,
                critical_threshold=0.95,
                duration_minutes=5,
                provider=ProviderType.GMAIL,
                comparison_operator="less_than"
            ),
            
            # Provider-specific Outlook thresholds
            MonitoringThreshold(
                metric=MonitoringMetric.BOUNCE_RATE,
                warning_threshold=0.03,
                critical_threshold=0.08,
                duration_minutes=5,
                provider=ProviderType.OUTLOOK,
                comparison_operator="greater_than"
            )
        ]
    
    @staticmethod
    def create_default_alert_rules() -> List[AlertRule]:
        """Create default alert rules"""
        return [
            AlertRule(
                rule_id="critical_delivery_rate_drop",
                name="Critical Delivery Rate Drop",
                description="Alert when delivery rate drops below critical threshold",
                conditions=[
                    {"metric": "delivery_rate", "operator": "less_than", "value": 0.90, "duration_minutes": 5}
                ],
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=30,
                notification_channels=["email", "slack", "sms"]
            ),
            
            AlertRule(
                rule_id="high_complaint_rate",
                name="High Complaint Rate",
                description="Alert when complaint rate exceeds acceptable levels",
                conditions=[
                    {"metric": "complaint_rate", "operator": "greater_than", "value": 0.002, "duration_minutes": 15}
                ],
                severity=AlertSeverity.WARNING,
                cooldown_minutes=60,
                notification_channels=["email", "slack"]
            ),
            
            AlertRule(
                rule_id="reputation_score_drop",
                name="Reputation Score Drop",
                description="Alert when sender reputation drops significantly",
                conditions=[
                    {"metric": "reputation_score", "operator": "less_than", "value": 0.60, "duration_minutes": 1}
                ],
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=15,
                notification_channels=["email", "slack"]
            ),
            
            AlertRule(
                rule_id="blacklist_detection",
                name="Blacklist Detection",
                description="Alert when IP appears on major blacklists",
                conditions=[
                    {"metric": "blacklist_status", "operator": "equals", "value": True}
                ],
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=5,
                notification_channels=["email", "slack", "sms"]
            ),
            
            AlertRule(
                rule_id="authentication_failure",
                name="Authentication Configuration Issues",
                description="Alert when SPF/DKIM/DMARC configuration issues are detected",
                conditions=[
                    {"metric": "authentication_status", "operator": "contains", "value": "missing"}
                ],
                severity=AlertSeverity.WARNING,
                cooldown_minutes=240,  # 4 hours
                notification_channels=["email"]
            )
        ]

# Usage demonstration and setup
async def demonstrate_deliverability_monitoring():
    """Demonstrate comprehensive deliverability monitoring system"""
    
    config = {
        'redis_url': 'redis://localhost:6379',
        'monitored_ips': ['192.168.1.100', '192.168.1.101'],
        'monitored_domains': ['example.com', 'marketing.example.com'],
        'dkim_selectors': ['default', 'selector1', 'k1'],
        'notification_channels': {
            'email': {
                'smtp_server': 'smtp.company.com',
                'username': 'alerts@company.com',
                'password': 'alert_password',
                'recipients': ['devops@company.com', 'marketing@company.com']
            },
            'slack': {
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            },
            'webhook': {
                'url': 'https://monitoring.company.com/webhook/alerts',
                'auth_header': 'Bearer webhook_token'
            }
        }
    }
    
    # Initialize monitoring system
    monitor = DeliverabilityMonitor(config)
    
    print("=== Email Deliverability Monitoring System Demo ===")
    
    # Set up default monitoring thresholds
    default_thresholds = MonitoringRuleBuilder.create_default_thresholds()
    for threshold in default_thresholds:
        await monitor.add_monitoring_threshold(threshold)
        print(f"Added threshold: {threshold.metric.value} for {threshold.provider.value if threshold.provider else 'all providers'}")
    
    # Set up default alert rules
    default_rules = MonitoringRuleBuilder.create_default_alert_rules()
    for rule in default_rules:
        await monitor.add_alert_rule(rule)
        print(f"Added alert rule: {rule.name}")
    
    # Wait for some monitoring cycles to collect data
    print("\nStarting monitoring cycles...")
    await asyncio.sleep(10)  # Let monitors run for a few cycles
    
    # Get dashboard data
    dashboard_data = await monitor.get_monitoring_dashboard_data()
    
    print(f"\n=== Monitoring Dashboard Data ===")
    print(f"System Status: {dashboard_data.get('system_status', 'unknown')}")
    print(f"Timestamp: {dashboard_data.get('timestamp')}")
    
    print(f"\nDelivery Metrics by Provider:")
    for provider, metrics in dashboard_data.get('delivery_metrics', {}).items():
        print(f"  {provider}:")
        print(f"    Delivery Rate: {metrics['delivery_rate']:.2%}")
        print(f"    Bounce Rate: {metrics['bounce_rate']:.2%}")
        print(f"    Complaint Rate: {metrics['complaint_rate']:.4%}")
        print(f"    Engagement Rate: {metrics['engagement_rate']:.2%}")
    
    print(f"\nActive Alerts: {len(dashboard_data.get('active_alerts', []))}")
    for alert in dashboard_data.get('active_alerts', [])[:5]:  # Show first 5
        print(f"  [{alert['severity']}] {alert['title']}")
    
    print(f"\nBlacklist Issues: {len(dashboard_data.get('blacklist_issues', []))}")
    for issue in dashboard_data.get('blacklist_issues', []):
        print(f"  IP {issue['ip_address']} on {issue['blacklist']} since {issue['first_detected']}")
    
    return monitor

if __name__ == "__main__":
    monitor = asyncio.run(demonstrate_deliverability_monitoring())
    
    print("\n=== Deliverability Monitoring System Active ===")
    print("Features:")
    print("  • Real-time reputation tracking across major providers")
    print("  • Comprehensive blacklist monitoring with DNS-based checks")
    print("  • Authentication status monitoring (SPF, DKIM, DMARC)")
    print("  • Delivery metrics analysis with statistical significance testing")
    print("  • Multi-channel alerting (email, Slack, SMS, webhooks)")
    print("  • Customizable thresholds and escalation rules")
    print("  • Historical trending and performance analytics")
    print("  • Automated incident response and notification routing")
```
{% endraw %}

## Advanced Analytics and Reporting

### Predictive Issue Detection

Implement machine learning algorithms that identify potential deliverability issues before they impact performance:

**Anomaly Detection Framework:**
```python
# Advanced anomaly detection for deliverability monitoring
class DeliverabilityAnomalyDetector:
    def __init__(self, monitor):
        self.monitor = monitor
        self.models = {}
        self.training_data = defaultdict(list)
        self.anomaly_thresholds = {
            'delivery_rate': 2.5,  # Standard deviations
            'bounce_rate': 2.0,
            'complaint_rate': 1.5,
            'engagement_rate': 2.0
        }
    
    async def detect_anomalies(self, metrics: DeliverabilityMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in deliverability metrics using statistical analysis"""
        
        anomalies = []
        provider_key = metrics.provider.value
        
        # Update training data
        self.training_data[provider_key].append({
            'timestamp': metrics.timestamp,
            'delivery_rate': metrics.delivery_rate,
            'bounce_rate': metrics.bounce_rate,
            'complaint_rate': metrics.complaint_rate,
            'engagement_rate': metrics.engagement_rate
        })
        
        # Keep only recent data for training
        cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        self.training_data[provider_key] = [
            d for d in self.training_data[provider_key] 
            if d['timestamp'] >= cutoff_time
        ]
        
        if len(self.training_data[provider_key]) < 100:  # Need sufficient data
            return anomalies
        
        # Check each metric for anomalies
        for metric_name in ['delivery_rate', 'bounce_rate', 'complaint_rate', 'engagement_rate']:
            anomaly = await self.check_metric_anomaly(
                provider_key, metric_name, getattr(metrics, metric_name)
            )
            
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    async def check_metric_anomaly(self, provider: str, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """Check if a metric value is anomalous compared to historical data"""
        
        historical_values = [
            d[metric] for d in self.training_data[provider]
            if metric in d
        ]
        
        if len(historical_values) < 100:
            return None
        
        # Calculate statistical measures
        mean_value = statistics.mean(historical_values)
        std_dev = statistics.stdev(historical_values)
        
        if std_dev == 0:  # No variation in historical data
            return None
        
        # Calculate z-score
        z_score = (value - mean_value) / std_dev
        threshold = self.anomaly_thresholds.get(metric, 2.0)
        
        if abs(z_score) > threshold:
            return {
                'provider': provider,
                'metric': metric,
                'current_value': value,
                'historical_mean': mean_value,
                'standard_deviation': std_dev,
                'z_score': z_score,
                'severity': 'critical' if abs(z_score) > threshold * 1.5 else 'warning',
                'anomaly_type': 'high' if z_score > 0 else 'low'
            }
        
        return None
```

### Performance Benchmarking

Compare deliverability performance against industry standards and historical baselines:

```python
class DeliverabilityBenchmarking:
    def __init__(self, monitor):
        self.monitor = monitor
        self.industry_benchmarks = {
            ProviderType.GMAIL: {
                'delivery_rate': {'excellent': 0.99, 'good': 0.97, 'average': 0.95, 'poor': 0.90},
                'bounce_rate': {'excellent': 0.01, 'good': 0.03, 'average': 0.05, 'poor': 0.10},
                'complaint_rate': {'excellent': 0.0005, 'good': 0.001, 'average': 0.002, 'poor': 0.005}
            },
            ProviderType.OUTLOOK: {
                'delivery_rate': {'excellent': 0.98, 'good': 0.96, 'average': 0.94, 'poor': 0.88},
                'bounce_rate': {'excellent': 0.015, 'good': 0.035, 'average': 0.06, 'poor': 0.12},
                'complaint_rate': {'excellent': 0.0008, 'good': 0.0015, 'average': 0.003, 'poor': 0.008}
            },
            ProviderType.YAHOO: {
                'delivery_rate': {'excellent': 0.97, 'good': 0.94, 'average': 0.91, 'poor': 0.85},
                'bounce_rate': {'excellent': 0.02, 'good': 0.04, 'average': 0.07, 'poor': 0.15},
                'complaint_rate': {'excellent': 0.001, 'good': 0.002, 'average': 0.004, 'poor': 0.010}
            }
        }
    
    async def generate_benchmark_report(self, time_period_days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive benchmarking report"""
        
        cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(days=time_period_days)
        cursor = self.monitor.db_conn.cursor()
        
        # Get average metrics for time period
        cursor.execute('''
            SELECT provider, AVG(delivery_rate), AVG(bounce_rate), AVG(complaint_rate), AVG(engagement_rate)
            FROM deliverability_metrics 
            WHERE timestamp >= ?
            GROUP BY provider
        ''', (cutoff_time,))
        
        report = {
            'time_period_days': time_period_days,
            'report_generated': datetime.datetime.utcnow().isoformat(),
            'provider_performance': {},
            'overall_grade': 'unknown',
            'recommendations': []
        }
        
        total_score = 0
        provider_count = 0
        
        for row in cursor.fetchall():
            provider, delivery_rate, bounce_rate, complaint_rate, engagement_rate = row
            provider_enum = ProviderType(provider)
            
            if provider_enum in self.industry_benchmarks:
                performance = self.analyze_provider_performance(
                    provider_enum, delivery_rate, bounce_rate, complaint_rate
                )
                
                report['provider_performance'][provider] = performance
                total_score += performance['overall_score']
                provider_count += 1
                
                # Add recommendations
                if performance['overall_score'] < 0.8:
                    report['recommendations'].extend(performance['recommendations'])
        
        # Calculate overall grade
        if provider_count > 0:
            avg_score = total_score / provider_count
            if avg_score >= 0.9:
                report['overall_grade'] = 'excellent'
            elif avg_score >= 0.8:
                report['overall_grade'] = 'good'
            elif avg_score >= 0.7:
                report['overall_grade'] = 'average'
            else:
                report['overall_grade'] = 'needs_improvement'
        
        return report
    
    def analyze_provider_performance(self, provider: ProviderType, delivery_rate: float, 
                                   bounce_rate: float, complaint_rate: float) -> Dict[str, Any]:
        """Analyze performance for specific provider against benchmarks"""
        
        benchmarks = self.industry_benchmarks[provider]
        performance = {
            'provider': provider.value,
            'metrics': {
                'delivery_rate': {
                    'value': delivery_rate,
                    'grade': self.grade_metric(delivery_rate, benchmarks['delivery_rate'], 'higher_better'),
                    'benchmark_comparison': self.compare_to_benchmark(delivery_rate, benchmarks['delivery_rate'], 'higher_better')
                },
                'bounce_rate': {
                    'value': bounce_rate,
                    'grade': self.grade_metric(bounce_rate, benchmarks['bounce_rate'], 'lower_better'),
                    'benchmark_comparison': self.compare_to_benchmark(bounce_rate, benchmarks['bounce_rate'], 'lower_better')
                },
                'complaint_rate': {
                    'value': complaint_rate,
                    'grade': self.grade_metric(complaint_rate, benchmarks['complaint_rate'], 'lower_better'),
                    'benchmark_comparison': self.compare_to_benchmark(complaint_rate, benchmarks['complaint_rate'], 'lower_better')
                }
            },
            'overall_score': 0,
            'recommendations': []
        }
        
        # Calculate overall score
        scores = []
        for metric_data in performance['metrics'].values():
            scores.append(self.grade_to_score(metric_data['grade']))
        
        performance['overall_score'] = sum(scores) / len(scores)
        
        # Generate recommendations
        performance['recommendations'] = self.generate_recommendations(performance['metrics'], provider)
        
        return performance
    
    def grade_metric(self, value: float, benchmarks: Dict[str, float], direction: str) -> str:
        """Grade a metric value against benchmarks"""
        
        if direction == 'higher_better':
            if value >= benchmarks['excellent']:
                return 'excellent'
            elif value >= benchmarks['good']:
                return 'good'
            elif value >= benchmarks['average']:
                return 'average'
            else:
                return 'poor'
        else:  # lower_better
            if value <= benchmarks['excellent']:
                return 'excellent'
            elif value <= benchmarks['good']:
                return 'good'
            elif value <= benchmarks['average']:
                return 'average'
            else:
                return 'poor'
```

## Implementation Best Practices

### 1. Scalable Infrastructure Design

**High-Performance Architecture:**
- Distributed monitoring across multiple geographic regions
- Load-balanced alert processing with automatic failover
- Efficient database indexing for fast metric retrieval
- Caching strategies for frequently accessed reputation data

### 2. Alert Fatigue Prevention

**Smart Alerting Strategies:**
- Intelligent alert grouping and deduplication
- Adaptive thresholds based on historical performance patterns  
- Escalation rules that prevent notification spam
- Contextual alerting that provides actionable insights

### 3. Data Retention and Privacy

**Compliance Framework:**
- Automated data retention policies with configurable timeframes
- Privacy-compliant logging that anonymizes sensitive information
- GDPR and CCPA compliance for stored monitoring data
- Secure API access with role-based authentication

### 4. Integration Capabilities

**Extensible Monitoring Platform:**
- Webhook integrations for custom notification systems
- API endpoints for external monitoring tool integration
- Export capabilities for business intelligence platforms
- Real-time streaming interfaces for dashboard applications

## Advanced Use Cases

### Multi-Tenant Monitoring

Support monitoring across multiple domains, IP ranges, and business units:

```python
class MultiTenantMonitor:
    def __init__(self, base_monitor):
        self.base_monitor = base_monitor
        self.tenant_configs = {}
        self.tenant_isolation = {}
    
    async def add_tenant(self, tenant_id: str, config: Dict[str, Any]):
        """Add monitoring configuration for new tenant"""
        
        self.tenant_configs[tenant_id] = {
            'monitored_ips': config.get('monitored_ips', []),
            'monitored_domains': config.get('monitored_domains', []),
            'notification_channels': config.get('notification_channels', {}),
            'custom_thresholds': config.get('custom_thresholds', []),
            'alert_rules': config.get('alert_rules', [])
        }
        
        # Set up tenant-specific monitoring
        await self.setup_tenant_monitoring(tenant_id)
    
    async def get_tenant_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get monitoring dashboard data filtered by tenant"""
        
        if tenant_id not in self.tenant_configs:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        tenant_config = self.tenant_configs[tenant_id]
        
        # Filter dashboard data by tenant's assets
        full_dashboard = await self.base_monitor.get_monitoring_dashboard_data()
        
        # Apply tenant filters
        filtered_dashboard = self.filter_dashboard_by_tenant(full_dashboard, tenant_config)
        
        return filtered_dashboard
```

### Automated Response Actions

Implement automatic remediation actions triggered by monitoring alerts:

```python
class AutomatedResponseEngine:
    def __init__(self, monitor):
        self.monitor = monitor
        self.response_actions = {
            'high_bounce_rate': self.pause_affected_campaigns,
            'blacklist_detection': self.initiate_delisting_process,
            'reputation_drop': self.switch_to_backup_ip,
            'authentication_failure': self.notify_dns_team
        }
    
    async def process_alert_response(self, alert: MonitoringAlert):
        """Execute automated responses based on alert type"""
        
        response_key = self.determine_response_key(alert)
        
        if response_key in self.response_actions:
            action = self.response_actions[response_key]
            
            try:
                await action(alert)
                self.monitor.logger.info(f"Executed automated response: {response_key}")
            except Exception as e:
                self.monitor.logger.error(f"Automated response failed: {str(e)}")
    
    async def pause_affected_campaigns(self, alert: MonitoringAlert):
        """Pause email campaigns for affected IPs/domains"""
        # Integration with campaign management system
        pass
    
    async def initiate_delisting_process(self, alert: MonitoringAlert):
        """Start blacklist removal process"""
        # Automated interaction with blacklist removal APIs
        pass
```

## Conclusion

Comprehensive email deliverability monitoring and alerting systems transform reactive email operations into proactive reputation management programs that maintain optimal inbox placement and business performance. Organizations implementing sophisticated monitoring frameworks typically see 40-60% faster issue resolution times, 70-85% reduction in deliverability incidents, and significant improvements in overall email program ROI.

The key to monitoring success lies in building systems that provide actionable insights rather than overwhelming alert volumes. Effective monitoring combines real-time alerting with predictive analytics, enabling teams to address issues before they impact business metrics.

Modern email programs require monitoring infrastructure that scales with business growth, integrates with existing marketing technology stacks, and provides the visibility needed for data-driven optimization decisions. The frameworks and implementation strategies outlined in this guide provide the foundation for building monitoring systems that support reliable, high-performance email delivery.

Remember that monitoring effectiveness depends on having clean, verified email data as the foundation. Consider integrating [professional email verification services](/services/) into your monitoring workflows to ensure accurate deliverability measurement and reliable alert thresholds.

Success in deliverability monitoring requires both technical excellence and operational discipline. Teams must balance comprehensive coverage with alert fatigue prevention, implement automated responses while maintaining human oversight, and continuously refine monitoring rules based on evolving email landscape conditions and business requirements.

The investment in robust monitoring infrastructure pays significant dividends through improved sender reputation, reduced deliverability incidents, enhanced customer satisfaction, and ultimately, better business outcomes from email marketing investments.