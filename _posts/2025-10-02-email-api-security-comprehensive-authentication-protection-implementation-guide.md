---
layout: post
title: "Email API Security: Comprehensive Authentication and Protection Implementation Guide"
date: 2025-10-02 08:00:00 -0500
categories: email-api api-security authentication rate-limiting security-implementation developer-tools
excerpt: "Master email API security with advanced authentication protocols, comprehensive rate limiting strategies, and robust protection mechanisms. Learn to implement enterprise-grade security measures that protect against API abuse, ensure data integrity, and maintain compliance while optimizing performance for high-volume email operations."
---

# Email API Security: Comprehensive Authentication and Protection Implementation Guide

Email API security represents a critical foundation for modern email infrastructure, with security breaches in email systems potentially exposing sensitive customer data, enabling unauthorized access to email campaigns, and compromising entire marketing operations. Organizations implementing comprehensive API security frameworks typically prevent 95-99% of attempted security breaches while maintaining optimal performance for legitimate API consumers.

Modern email APIs face increasingly sophisticated attack vectors, from credential stuffing and token hijacking to advanced persistent threats targeting email infrastructure. The complexity of modern email ecosystems, spanning multiple providers, authentication methods, and integration points, demands robust security architecture that addresses both traditional vulnerabilities and emerging threat patterns.

This comprehensive guide explores advanced API security implementation strategies, authentication protocols, and protection mechanisms that enable development teams, security engineers, and product managers to build resilient email APIs that maintain security while supporting high-volume operations and diverse integration requirements.

## Advanced Authentication Architecture

### Multi-Layer Authentication Framework

Effective API security requires sophisticated authentication mechanisms that adapt to various use cases and threat levels:

**Primary Authentication Methods:**
- API key-based authentication with rotation and scope management
- OAuth 2.0 implementation with proper token handling and refresh mechanisms  
- JWT (JSON Web Token) authentication with signature verification
- mTLS (mutual TLS) for high-security enterprise integrations

**Secondary Authentication Factors:**
- IP address allowlisting with dynamic range management
- Request signature verification using HMAC-SHA256 algorithms
- Timestamp-based request validation to prevent replay attacks
- Device fingerprinting for suspicious pattern detection

**Context-Aware Authentication:**
- Geographic location validation and anomaly detection
- Request pattern analysis for behavioral authentication
- Risk-based authentication escalation based on request sensitivity
- Dynamic authentication requirements based on API endpoint classification

### Comprehensive API Security Implementation

Build robust authentication and protection systems for email APIs:

{% raw %}
```python
# Advanced email API security and authentication system
import asyncio
import jwt
import hmac
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis
import bcrypt
import ipaddress
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import aiohttp
import ssl
from functools import wraps
import time
import json
from collections import defaultdict, deque

class AuthMethod(Enum):
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    MTLS = "mtls"
    HMAC_SIGNATURE = "hmac_signature"

class SecurityLevel(Enum):
    PUBLIC = "public"
    STANDARD = "standard"  
    SENSITIVE = "sensitive"
    CRITICAL = "critical"

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIClient:
    client_id: str
    client_secret: str
    name: str
    allowed_endpoints: List[str]
    rate_limits: Dict[str, int]
    security_level: SecurityLevel
    allowed_ips: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    event_id: str
    event_type: str
    client_id: str
    ip_address: str
    endpoint: str
    threat_level: ThreatLevel
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False

@dataclass
class RateLimitRule:
    endpoint_pattern: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    window_size: int = 60  # seconds

class EmailAPISecurityManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0)
        )
        
        # Authentication components
        self.jwt_secret = config.get('jwt_secret', secrets.token_urlsafe(32))
        self.api_clients = {}
        self.revoked_tokens = set()
        
        # Security monitoring
        self.security_events = []
        self.suspicious_ips = set()
        self.failed_auth_attempts = defaultdict(deque)
        
        # Rate limiting
        self.rate_limiters = {}
        self.default_rate_limits = config.get('default_rate_limits', {
            'requests_per_minute': 100,
            'requests_per_hour': 1000,
            'requests_per_day': 10000
        })
        
        # Certificate management for mTLS
        self.ca_cert = None
        self.server_cert = None
        self.server_key = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize security components
        self.initialize_security_infrastructure()
    
    def initialize_security_infrastructure(self):
        """Initialize security infrastructure components"""
        try:
            # Load or generate certificates for mTLS
            self.setup_mtls_certificates()
            
            # Initialize rate limiting infrastructure
            self.setup_rate_limiting()
            
            # Setup security monitoring
            self.setup_security_monitoring()
            
            # Load existing API clients
            self.load_api_clients()
            
            self.logger.info("Email API security infrastructure initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security infrastructure: {str(e)}")
            raise
    
    def setup_mtls_certificates(self):
        """Setup mutual TLS certificates for high-security authentication"""
        try:
            cert_config = self.config.get('mtls_config', {})
            
            if cert_config.get('ca_cert_path'):
                with open(cert_config['ca_cert_path'], 'rb') as f:
                    self.ca_cert = f.read()
            
            if cert_config.get('server_cert_path'):
                with open(cert_config['server_cert_path'], 'rb') as f:
                    self.server_cert = f.read()
                    
            if cert_config.get('server_key_path'):
                with open(cert_config['server_key_path'], 'rb') as f:
                    self.server_key = f.read()
            
            self.logger.info("mTLS certificates loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"mTLS certificate setup failed: {str(e)}")
    
    def setup_rate_limiting(self):
        """Setup rate limiting infrastructure"""
        rate_limit_rules = self.config.get('rate_limit_rules', [])
        
        for rule_config in rate_limit_rules:
            rule = RateLimitRule(
                endpoint_pattern=rule_config['endpoint_pattern'],
                requests_per_minute=rule_config.get('requests_per_minute', 100),
                requests_per_hour=rule_config.get('requests_per_hour', 1000),
                requests_per_day=rule_config.get('requests_per_day', 10000),
                burst_limit=rule_config.get('burst_limit', 10),
                window_size=rule_config.get('window_size', 60)
            )
            self.rate_limiters[rule.endpoint_pattern] = rule
        
        self.logger.info(f"Rate limiting configured for {len(rate_limit_rules)} endpoint patterns")
    
    def setup_security_monitoring(self):
        """Setup security event monitoring and alerting"""
        self.security_config = self.config.get('security_monitoring', {})
        
        # Configure threat detection thresholds
        self.threat_thresholds = {
            'failed_auth_attempts': self.security_config.get('failed_auth_threshold', 5),
            'suspicious_request_rate': self.security_config.get('suspicious_rate_threshold', 1000),
            'unusual_endpoint_access': self.security_config.get('unusual_endpoint_threshold', 50),
            'geographic_anomaly_score': self.security_config.get('geo_anomaly_threshold', 0.8)
        }
        
        self.logger.info("Security monitoring configured")
    
    async def authenticate_request(self, request: Dict[str, Any]) -> Tuple[bool, Optional[APIClient], Optional[str]]:
        """Comprehensive request authentication with multiple methods"""
        try:
            client_ip = request.get('client_ip')
            endpoint = request.get('endpoint')
            headers = request.get('headers', {})
            
            # Check if IP is blocked
            if client_ip in self.suspicious_ips:
                await self.log_security_event(
                    event_type="blocked_ip_attempt",
                    client_id="unknown",
                    ip_address=client_ip,
                    endpoint=endpoint,
                    threat_level=ThreatLevel.HIGH,
                    details={"reason": "IP address blocked"}
                )
                return False, None, "IP address blocked"
            
            # Try different authentication methods
            auth_methods = [
                self.authenticate_oauth2,
                self.authenticate_jwt,
                self.authenticate_api_key,
                self.authenticate_hmac_signature
            ]
            
            for auth_method in auth_methods:
                success, client, error = await auth_method(request)
                if success:
                    # Additional security checks
                    if not await self.validate_client_access(client, endpoint, client_ip):
                        return False, None, "Access denied"
                    
                    # Check rate limits
                    if not await self.check_rate_limits(client, endpoint):
                        return False, client, "Rate limit exceeded"
                    
                    # Update client usage
                    await self.update_client_usage(client, endpoint, client_ip)
                    
                    return True, client, None
            
            # All authentication methods failed
            await self.handle_failed_authentication(client_ip, endpoint, headers)
            return False, None, "Authentication failed"
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False, None, "Authentication system error"
    
    async def authenticate_api_key(self, request: Dict[str, Any]) -> Tuple[bool, Optional[APIClient], Optional[str]]:
        """Authenticate using API key"""
        try:
            headers = request.get('headers', {})
            api_key = headers.get('X-API-Key') or headers.get('Authorization', '').replace('Bearer ', '')
            
            if not api_key:
                return False, None, "No API key provided"
            
            # Find client by API key
            client = await self.find_client_by_api_key(api_key)
            if not client:
                return False, None, "Invalid API key"
            
            if not client.is_active:
                return False, None, "Client account inactive"
            
            return True, client, None
            
        except Exception as e:
            self.logger.error(f"API key authentication error: {str(e)}")
            return False, None, "API key authentication failed"
    
    async def authenticate_oauth2(self, request: Dict[str, Any]) -> Tuple[bool, Optional[APIClient], Optional[str]]:
        """Authenticate using OAuth 2.0 access token"""
        try:
            headers = request.get('headers', {})
            auth_header = headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                return False, None, "Invalid OAuth2 token format"
            
            access_token = auth_header.replace('Bearer ', '')
            
            # Validate access token
            token_data = await self.validate_oauth2_token(access_token)
            if not token_data:
                return False, None, "Invalid or expired access token"
            
            client = await self.find_client_by_id(token_data['client_id'])
            if not client:
                return False, None, "Client not found"
            
            return True, client, None
            
        except Exception as e:
            self.logger.error(f"OAuth2 authentication error: {str(e)}")
            return False, None, "OAuth2 authentication failed"
    
    async def authenticate_jwt(self, request: Dict[str, Any]) -> Tuple[bool, Optional[APIClient], Optional[str]]:
        """Authenticate using JWT token"""
        try:
            headers = request.get('headers', {})
            auth_header = headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                return False, None, "Invalid JWT token format"
            
            token = auth_header.replace('Bearer ', '')
            
            # Check if token is revoked
            if token in self.revoked_tokens:
                return False, None, "Token has been revoked"
            
            # Decode and validate JWT
            try:
                payload = jwt.decode(
                    token, 
                    self.jwt_secret, 
                    algorithms=['HS256']
                )
            except jwt.ExpiredSignatureError:
                return False, None, "Token has expired"
            except jwt.InvalidTokenError:
                return False, None, "Invalid token"
            
            # Validate token claims
            if not self.validate_jwt_claims(payload, request):
                return False, None, "Invalid token claims"
            
            client = await self.find_client_by_id(payload['client_id'])
            if not client:
                return False, None, "Client not found"
            
            return True, client, None
            
        except Exception as e:
            self.logger.error(f"JWT authentication error: {str(e)}")
            return False, None, "JWT authentication failed"
    
    async def authenticate_hmac_signature(self, request: Dict[str, Any]) -> Tuple[bool, Optional[APIClient], Optional[str]]:
        """Authenticate using HMAC signature"""
        try:
            headers = request.get('headers', {})
            signature = headers.get('X-Signature')
            timestamp = headers.get('X-Timestamp')
            client_id = headers.get('X-Client-Id')
            
            if not all([signature, timestamp, client_id]):
                return False, None, "Missing HMAC signature components"
            
            # Check timestamp to prevent replay attacks
            request_time = datetime.fromtimestamp(int(timestamp))
            if abs((datetime.utcnow() - request_time).total_seconds()) > 300:  # 5 minute window
                return False, None, "Request timestamp outside valid window"
            
            client = await self.find_client_by_id(client_id)
            if not client:
                return False, None, "Client not found"
            
            # Verify HMAC signature
            request_body = request.get('body', '')
            expected_signature = self.generate_hmac_signature(
                client.client_secret,
                request.get('method', 'GET'),
                request.get('endpoint', ''),
                request_body,
                timestamp
            )
            
            if not hmac.compare_digest(signature, expected_signature):
                return False, None, "Invalid HMAC signature"
            
            return True, client, None
            
        except Exception as e:
            self.logger.error(f"HMAC authentication error: {str(e)}")
            return False, None, "HMAC authentication failed"
    
    def generate_hmac_signature(self, secret: str, method: str, endpoint: str, body: str, timestamp: str) -> str:
        """Generate HMAC-SHA256 signature for request"""
        message = f"{method}\n{endpoint}\n{body}\n{timestamp}"
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def validate_oauth2_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Validate OAuth2 access token"""
        try:
            # Check token in Redis cache
            cached_token = self.redis_client.get(f"oauth2_token:{access_token}")
            if cached_token:
                return json.loads(cached_token)
            
            # If not cached, validate with OAuth2 provider
            # This would typically involve calling the OAuth2 introspection endpoint
            # For this example, we'll simulate token validation
            
            return None
            
        except Exception as e:
            self.logger.error(f"OAuth2 token validation error: {str(e)}")
            return None
    
    def validate_jwt_claims(self, payload: Dict[str, Any], request: Dict[str, Any]) -> bool:
        """Validate JWT token claims"""
        try:
            # Check required claims
            required_claims = ['client_id', 'exp', 'iat', 'aud']
            for claim in required_claims:
                if claim not in payload:
                    return False
            
            # Validate audience
            if payload['aud'] != self.config.get('jwt_audience', 'email-api'):
                return False
            
            # Additional custom validations can be added here
            
            return True
            
        except Exception as e:
            self.logger.error(f"JWT claims validation error: {str(e)}")
            return False
    
    async def validate_client_access(self, client: APIClient, endpoint: str, client_ip: str) -> bool:
        """Validate client access permissions"""
        try:
            # Check if client is active
            if not client.is_active:
                return False
            
            # Check endpoint permissions
            if client.allowed_endpoints and endpoint not in client.allowed_endpoints:
                if not any(endpoint.startswith(pattern.rstrip('*')) for pattern in client.allowed_endpoints if pattern.endswith('*')):
                    return False
            
            # Check IP allowlist
            if client.allowed_ips:
                ip_allowed = False
                for allowed_ip in client.allowed_ips:
                    try:
                        if ipaddress.ip_address(client_ip) in ipaddress.ip_network(allowed_ip, strict=False):
                            ip_allowed = True
                            break
                    except ipaddress.AddressValueError:
                        # Handle individual IP addresses
                        if client_ip == allowed_ip:
                            ip_allowed = True
                            break
                
                if not ip_allowed:
                    await self.log_security_event(
                        event_type="ip_access_denied",
                        client_id=client.client_id,
                        ip_address=client_ip,
                        endpoint=endpoint,
                        threat_level=ThreatLevel.MEDIUM,
                        details={"allowed_ips": client.allowed_ips}
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Client access validation error: {str(e)}")
            return False
    
    async def check_rate_limits(self, client: APIClient, endpoint: str) -> bool:
        """Check if request exceeds rate limits"""
        try:
            current_time = int(time.time())
            
            # Get rate limit rules for endpoint
            rate_rule = self.get_rate_limit_rule(endpoint)
            client_limits = client.rate_limits or {}
            
            # Check per-minute limit
            minute_key = f"rate_limit:{client.client_id}:{endpoint}:minute:{current_time // 60}"
            current_minute_requests = self.redis_client.incr(minute_key)
            self.redis_client.expire(minute_key, 120)  # Expire after 2 minutes
            
            minute_limit = client_limits.get('requests_per_minute', rate_rule.requests_per_minute)
            if current_minute_requests > minute_limit:
                await self.log_security_event(
                    event_type="rate_limit_exceeded",
                    client_id=client.client_id,
                    ip_address="",
                    endpoint=endpoint,
                    threat_level=ThreatLevel.MEDIUM,
                    details={
                        "limit_type": "per_minute",
                        "limit": minute_limit,
                        "current": current_minute_requests
                    }
                )
                return False
            
            # Check per-hour limit
            hour_key = f"rate_limit:{client.client_id}:{endpoint}:hour:{current_time // 3600}"
            current_hour_requests = self.redis_client.incr(hour_key)
            self.redis_client.expire(hour_key, 7200)  # Expire after 2 hours
            
            hour_limit = client_limits.get('requests_per_hour', rate_rule.requests_per_hour)
            if current_hour_requests > hour_limit:
                return False
            
            # Check per-day limit
            day_key = f"rate_limit:{client.client_id}:{endpoint}:day:{current_time // 86400}"
            current_day_requests = self.redis_client.incr(day_key)
            self.redis_client.expire(day_key, 172800)  # Expire after 2 days
            
            day_limit = client_limits.get('requests_per_day', rate_rule.requests_per_day)
            if current_day_requests > day_limit:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {str(e)}")
            return True  # Allow request if rate limit check fails
    
    def get_rate_limit_rule(self, endpoint: str) -> RateLimitRule:
        """Get rate limit rule for endpoint"""
        for pattern, rule in self.rate_limiters.items():
            if endpoint.startswith(pattern.rstrip('*')):
                return rule
        
        # Return default rule if no specific rule found
        return RateLimitRule(
            endpoint_pattern="*",
            requests_per_minute=self.default_rate_limits['requests_per_minute'],
            requests_per_hour=self.default_rate_limits['requests_per_hour'],
            requests_per_day=self.default_rate_limits['requests_per_day'],
            burst_limit=10
        )
    
    async def handle_failed_authentication(self, client_ip: str, endpoint: str, headers: Dict[str, Any]):
        """Handle failed authentication attempts"""
        try:
            current_time = datetime.utcnow()
            
            # Track failed attempts by IP
            self.failed_auth_attempts[client_ip].append(current_time)
            
            # Remove attempts older than 1 hour
            cutoff_time = current_time - timedelta(hours=1)
            while self.failed_auth_attempts[client_ip] and self.failed_auth_attempts[client_ip][0] < cutoff_time:
                self.failed_auth_attempts[client_ip].popleft()
            
            # Check if IP should be temporarily blocked
            if len(self.failed_auth_attempts[client_ip]) >= self.threat_thresholds['failed_auth_attempts']:
                self.suspicious_ips.add(client_ip)
                
                await self.log_security_event(
                    event_type="ip_temporarily_blocked",
                    client_id="unknown",
                    ip_address=client_ip,
                    endpoint=endpoint,
                    threat_level=ThreatLevel.HIGH,
                    details={
                        "failed_attempts": len(self.failed_auth_attempts[client_ip]),
                        "headers": headers
                    }
                )
                
                # Schedule IP unblocking after cooldown period
                asyncio.create_task(self.unblock_ip_after_cooldown(client_ip))
            
        except Exception as e:
            self.logger.error(f"Failed authentication handling error: {str(e)}")
    
    async def unblock_ip_after_cooldown(self, client_ip: str):
        """Unblock IP address after cooldown period"""
        cooldown_minutes = self.config.get('ip_cooldown_minutes', 60)
        await asyncio.sleep(cooldown_minutes * 60)
        
        if client_ip in self.suspicious_ips:
            self.suspicious_ips.remove(client_ip)
            self.logger.info(f"IP address {client_ip} unblocked after cooldown")
    
    async def log_security_event(self, event_type: str, client_id: str, ip_address: str, 
                                endpoint: str, threat_level: ThreatLevel, details: Dict[str, Any]):
        """Log security events for monitoring and analysis"""
        try:
            event = SecurityEvent(
                event_id=hashlib.md5(f"{event_type}{client_id}{ip_address}{datetime.utcnow()}".encode()).hexdigest()[:12],
                event_type=event_type,
                client_id=client_id,
                ip_address=ip_address,
                endpoint=endpoint,
                threat_level=threat_level,
                timestamp=datetime.utcnow(),
                details=details
            )
            
            self.security_events.append(event)
            
            # Store in Redis for persistence
            event_data = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'client_id': event.client_id,
                'ip_address': event.ip_address,
                'endpoint': event.endpoint,
                'threat_level': event.threat_level.value,
                'timestamp': event.timestamp.isoformat(),
                'details': event.details
            }
            
            self.redis_client.lpush('security_events', json.dumps(event_data))
            self.redis_client.ltrim('security_events', 0, 9999)  # Keep last 10k events
            
            # Send alerts for high-severity events
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                await self.send_security_alert(event)
            
        except Exception as e:
            self.logger.error(f"Security event logging error: {str(e)}")
    
    async def send_security_alert(self, event: SecurityEvent):
        """Send security alert for high-severity events"""
        try:
            alert_config = self.config.get('security_alerts', {})
            
            if alert_config.get('webhook_url'):
                alert_data = {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'threat_level': event.threat_level.value,
                    'client_id': event.client_id,
                    'ip_address': event.ip_address,
                    'endpoint': event.endpoint,
                    'timestamp': event.timestamp.isoformat(),
                    'details': event.details
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        alert_config['webhook_url'],
                        json=alert_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Security alert sent for event {event.event_id}")
                        else:
                            self.logger.warning(f"Failed to send security alert: HTTP {response.status}")
            
        except Exception as e:
            self.logger.error(f"Security alert sending error: {str(e)}")
    
    async def create_api_client(self, name: str, allowed_endpoints: List[str], 
                              security_level: SecurityLevel, **kwargs) -> APIClient:
        """Create new API client with security credentials"""
        try:
            client_id = kwargs.get('client_id', f"client_{secrets.token_urlsafe(16)}")
            client_secret = kwargs.get('client_secret', secrets.token_urlsafe(32))
            
            # Generate API keys
            api_keys = [f"ak_{secrets.token_urlsafe(32)}" for _ in range(kwargs.get('num_api_keys', 1))]
            
            client = APIClient(
                client_id=client_id,
                client_secret=client_secret,
                name=name,
                allowed_endpoints=allowed_endpoints,
                rate_limits=kwargs.get('rate_limits', {}),
                security_level=security_level,
                allowed_ips=kwargs.get('allowed_ips', []),
                api_keys=api_keys,
                metadata=kwargs.get('metadata', {})
            )
            
            self.api_clients[client_id] = client
            
            # Store in Redis for persistence
            await self.store_client(client)
            
            self.logger.info(f"Created API client: {client_id} ({name})")
            return client
            
        except Exception as e:
            self.logger.error(f"API client creation error: {str(e)}")
            raise
    
    async def store_client(self, client: APIClient):
        """Store API client data in Redis"""
        try:
            client_data = {
                'client_id': client.client_id,
                'client_secret': client.client_secret,
                'name': client.name,
                'allowed_endpoints': client.allowed_endpoints,
                'rate_limits': client.rate_limits,
                'security_level': client.security_level.value,
                'allowed_ips': client.allowed_ips,
                'api_keys': client.api_keys,
                'created_at': client.created_at.isoformat(),
                'last_used': client.last_used.isoformat() if client.last_used else None,
                'is_active': client.is_active,
                'metadata': client.metadata
            }
            
            self.redis_client.hset('api_clients', client.client_id, json.dumps(client_data))
            
            # Index API keys for quick lookup
            for api_key in client.api_keys:
                self.redis_client.set(f"api_key:{api_key}", client.client_id, ex=86400*365)  # 1 year expiry
            
        except Exception as e:
            self.logger.error(f"Client storage error: {str(e)}")
    
    async def find_client_by_api_key(self, api_key: str) -> Optional[APIClient]:
        """Find API client by API key"""
        try:
            client_id = self.redis_client.get(f"api_key:{api_key}")
            if client_id:
                return await self.find_client_by_id(client_id.decode('utf-8'))
            return None
            
        except Exception as e:
            self.logger.error(f"Client lookup by API key error: {str(e)}")
            return None
    
    async def find_client_by_id(self, client_id: str) -> Optional[APIClient]:
        """Find API client by client ID"""
        try:
            if client_id in self.api_clients:
                return self.api_clients[client_id]
            
            # Try loading from Redis
            client_data = self.redis_client.hget('api_clients', client_id)
            if client_data:
                data = json.loads(client_data)
                client = APIClient(
                    client_id=data['client_id'],
                    client_secret=data['client_secret'],
                    name=data['name'],
                    allowed_endpoints=data['allowed_endpoints'],
                    rate_limits=data['rate_limits'],
                    security_level=SecurityLevel(data['security_level']),
                    allowed_ips=data['allowed_ips'],
                    api_keys=data['api_keys'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None,
                    is_active=data['is_active'],
                    metadata=data['metadata']
                )
                
                self.api_clients[client_id] = client
                return client
            
            return None
            
        except Exception as e:
            self.logger.error(f"Client lookup by ID error: {str(e)}")
            return None
    
    def load_api_clients(self):
        """Load existing API clients from storage"""
        try:
            client_ids = self.redis_client.hkeys('api_clients')
            loaded_count = 0
            
            for client_id in client_ids:
                client_data = self.redis_client.hget('api_clients', client_id)
                if client_data:
                    try:
                        data = json.loads(client_data)
                        client = APIClient(
                            client_id=data['client_id'],
                            client_secret=data['client_secret'],
                            name=data['name'],
                            allowed_endpoints=data['allowed_endpoints'],
                            rate_limits=data['rate_limits'],
                            security_level=SecurityLevel(data['security_level']),
                            allowed_ips=data['allowed_ips'],
                            api_keys=data['api_keys'],
                            created_at=datetime.fromisoformat(data['created_at']),
                            last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None,
                            is_active=data['is_active'],
                            metadata=data['metadata']
                        )
                        
                        self.api_clients[client_id.decode('utf-8')] = client
                        loaded_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load client {client_id}: {str(e)}")
            
            self.logger.info(f"Loaded {loaded_count} API clients")
            
        except Exception as e:
            self.logger.error(f"API client loading error: {str(e)}")
    
    async def update_client_usage(self, client: APIClient, endpoint: str, ip_address: str):
        """Update client usage statistics"""
        try:
            client.last_used = datetime.utcnow()
            
            # Update usage statistics
            current_date = datetime.utcnow().strftime('%Y-%m-%d')
            usage_key = f"usage:{client.client_id}:{current_date}"
            
            usage_data = {
                'endpoint': endpoint,
                'ip_address': ip_address,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.redis_client.lpush(usage_key, json.dumps(usage_data))
            self.redis_client.expire(usage_key, 86400 * 30)  # Keep for 30 days
            
            # Update client in storage
            await self.store_client(client)
            
        except Exception as e:
            self.logger.error(f"Client usage update error: {str(e)}")
    
    def generate_jwt_token(self, client: APIClient, expires_in: int = 3600) -> str:
        """Generate JWT token for client"""
        try:
            payload = {
                'client_id': client.client_id,
                'aud': self.config.get('jwt_audience', 'email-api'),
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=expires_in),
                'scope': client.allowed_endpoints
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            return token
            
        except Exception as e:
            self.logger.error(f"JWT generation error: {str(e)}")
            raise
    
    async def revoke_token(self, token: str):
        """Revoke JWT token"""
        try:
            # Add to revoked tokens set
            self.revoked_tokens.add(token)
            
            # Store in Redis with expiration
            self.redis_client.set(f"revoked_token:{token}", "1", ex=86400)  # 24 hour expiry
            
            self.logger.info("Token revoked successfully")
            
        except Exception as e:
            self.logger.error(f"Token revocation error: {str(e)}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        try:
            current_time = datetime.utcnow()
            
            # Analyze recent security events
            recent_events = [
                event for event in self.security_events
                if (current_time - event.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            # Categorize events by type and threat level
            event_summary = defaultdict(lambda: defaultdict(int))
            for event in recent_events:
                event_summary[event.event_type][event.threat_level.value] += 1
            
            # Calculate security metrics
            total_failed_auths = sum(len(attempts) for attempts in self.failed_auth_attempts.values())
            blocked_ips_count = len(self.suspicious_ips)
            active_clients = len([c for c in self.api_clients.values() if c.is_active])
            
            report = {
                'timestamp': current_time.isoformat(),
                'summary': {
                    'total_active_clients': active_clients,
                    'total_failed_auth_attempts': total_failed_auths,
                    'blocked_ips': blocked_ips_count,
                    'recent_security_events': len(recent_events)
                },
                'event_breakdown': dict(event_summary),
                'top_threat_sources': self.get_top_threat_sources(),
                'security_recommendations': self.generate_security_recommendations(),
                'system_health': {
                    'authentication_success_rate': self.calculate_auth_success_rate(),
                    'rate_limit_violations': self.count_rate_limit_violations(),
                    'unusual_activity_detected': self.detect_unusual_activity()
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Security report generation error: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_top_threat_sources(self) -> List[Dict[str, Any]]:
        """Get top threat sources by IP address"""
        try:
            ip_threat_counts = defaultdict(int)
            
            for event in self.security_events[-1000:]:  # Last 1000 events
                if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    ip_threat_counts[event.ip_address] += 1
            
            top_sources = sorted(
                ip_threat_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return [
                {'ip_address': ip, 'threat_count': count}
                for ip, count in top_sources
            ]
            
        except Exception as e:
            self.logger.error(f"Top threat sources analysis error: {str(e)}")
            return []
    
    def generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state"""
        recommendations = []
        
        try:
            # Check for high failed authentication rates
            if len(self.failed_auth_attempts) > 50:
                recommendations.append("Consider implementing CAPTCHA for repeated failed authentication attempts")
            
            # Check for blocked IPs
            if len(self.suspicious_ips) > 10:
                recommendations.append("Review and update IP allowlists for legitimate clients")
            
            # Check for inactive clients
            inactive_clients = [c for c in self.api_clients.values() if not c.is_active]
            if len(inactive_clients) > len(self.api_clients) * 0.3:
                recommendations.append("Clean up inactive API clients to reduce attack surface")
            
            # Check authentication methods
            if not self.config.get('mtls_enabled', False):
                recommendations.append("Consider enabling mTLS for high-security client authentication")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Security recommendations generation error: {str(e)}")
            return ["Unable to generate recommendations due to system error"]

# Security middleware and decorators
def require_auth(security_level: SecurityLevel = SecurityLevel.STANDARD):
    """Decorator to require authentication for API endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with your web framework (FastAPI, Flask, etc.)
            # to extract request data and perform authentication
            
            request_data = {
                'headers': kwargs.get('headers', {}),
                'client_ip': kwargs.get('client_ip', ''),
                'endpoint': kwargs.get('endpoint', ''),
                'method': kwargs.get('method', 'GET'),
                'body': kwargs.get('body', '')
            }
            
            security_manager = kwargs.get('security_manager')
            if not security_manager:
                raise ValueError("Security manager not provided")
            
            success, client, error = await security_manager.authenticate_request(request_data)
            
            if not success:
                return {'error': error, 'status_code': 401}
            
            if client.security_level.value < security_level.value:
                return {'error': 'Insufficient security level', 'status_code': 403}
            
            # Add client info to kwargs
            kwargs['authenticated_client'] = client
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Usage example
async def main():
    """Example usage of email API security system"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'jwt_secret': 'your-jwt-secret-key',
        'jwt_audience': 'email-api',
        'default_rate_limits': {
            'requests_per_minute': 100,
            'requests_per_hour': 1000,
            'requests_per_day': 10000
        },
        'rate_limit_rules': [
            {
                'endpoint_pattern': '/api/send/*',
                'requests_per_minute': 50,
                'requests_per_hour': 500,
                'requests_per_day': 5000,
                'burst_limit': 10
            },
            {
                'endpoint_pattern': '/api/validate/*',
                'requests_per_minute': 200,
                'requests_per_hour': 2000,
                'requests_per_day': 20000,
                'burst_limit': 20
            }
        ],
        'security_monitoring': {
            'failed_auth_threshold': 5,
            'suspicious_rate_threshold': 1000,
            'unusual_endpoint_threshold': 50
        },
        'security_alerts': {
            'webhook_url': 'https://alerts.yourcompany.com/webhook'
        },
        'ip_cooldown_minutes': 60
    }
    
    # Initialize security manager
    security_manager = EmailAPISecurityManager(config)
    
    try:
        # Create API clients for different use cases
        marketing_client = await security_manager.create_api_client(
            name="Marketing Team",
            allowed_endpoints=["/api/send/*", "/api/campaigns/*"],
            security_level=SecurityLevel.STANDARD,
            allowed_ips=["192.168.1.0/24", "10.0.0.100"],
            rate_limits={
                'requests_per_minute': 200,
                'requests_per_hour': 2000,
                'requests_per_day': 20000
            }
        )
        
        analytics_client = await security_manager.create_api_client(
            name="Analytics Service",
            allowed_endpoints=["/api/analytics/*", "/api/reports/*"],
            security_level=SecurityLevel.SENSITIVE,
            num_api_keys=2,
            metadata={"service_type": "internal", "department": "analytics"}
        )
        
        print(f"Marketing Client ID: {marketing_client.client_id}")
        print(f"Marketing API Keys: {marketing_client.api_keys}")
        
        print(f"Analytics Client ID: {analytics_client.client_id}")
        print(f"Analytics API Keys: {analytics_client.api_keys}")
        
        # Generate JWT token for client
        jwt_token = security_manager.generate_jwt_token(marketing_client, expires_in=7200)
        print(f"JWT Token: {jwt_token}")
        
        # Example authentication test
        test_request = {
            'headers': {'X-API-Key': marketing_client.api_keys[0]},
            'client_ip': '192.168.1.50',
            'endpoint': '/api/send/bulk',
            'method': 'POST',
            'body': '{"recipients": ["test@example.com"], "subject": "Test"}'
        }
        
        success, client, error = await security_manager.authenticate_request(test_request)
        print(f"Authentication result: {success}, Client: {client.name if client else None}, Error: {error}")
        
        # Generate security report
        security_report = security_manager.get_security_report()
        print("Security Report:", json.dumps(security_report, indent=2, default=str))
        
    except Exception as e:
        print(f"Security system error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
```
{% endraw %}

## Rate Limiting and Abuse Prevention

### Advanced Rate Limiting Strategies

Implement sophisticated rate limiting mechanisms that balance security with usability:

**Dynamic Rate Limiting:**
- Behavior-based rate adjustment based on client reputation
- Time-of-day and traffic pattern-aware limiting
- Endpoint-specific rate limits based on resource intensity
- Burst allowances for legitimate traffic spikes

**Distributed Rate Limiting:**
- Cross-server rate limit synchronization using Redis
- Geographic rate limiting for global API deployments
- Client-specific rate limit pools with borrowing mechanisms
- Hierarchical rate limiting for organization-level controls

**Adaptive Abuse Detection:**
- Machine learning-based anomaly detection for API usage patterns
- Behavioral fingerprinting for automated bot detection
- Progressive penalty systems for policy violations
- Automated temporary restrictions with escalating responses

### Protection Against Common Attack Vectors

Implement comprehensive protection against email API-specific attacks:

{% raw %}
```javascript
// Advanced API abuse prevention and monitoring system
class EmailAPIAbusePreventionSystem {
    constructor(config) {
        this.config = config;
        this.redis = new Redis(config.redis);
        this.suspiciousPatterns = new Map();
        this.clientBehaviorProfiles = new Map();
        this.alertManager = new AlertManager(config.alerting);
        
        // Attack pattern definitions
        this.attackPatterns = {
            credential_stuffing: {
                indicators: ['high_failure_rate', 'distributed_ips', 'common_passwords'],
                threshold_score: 0.8,
                mitigation: 'temp_block_with_captcha'
            },
            spam_injection: {
                indicators: ['bulk_sending_spikes', 'low_engagement', 'spam_content'],
                threshold_score: 0.7,
                mitigation: 'content_review_required'
            },
            data_harvesting: {
                indicators: ['validation_only_usage', 'sequential_scanning', 'high_volume'],
                threshold_score: 0.6,
                mitigation: 'rate_limit_reduction'
            },
            api_scraping: {
                indicators: ['rapid_requests', 'automated_patterns', 'unusual_endpoints'],
                threshold_score: 0.75,
                mitigation: 'progressive_delays'
            }
        };
        
        this.initialize();
    }
    
    async initialize() {
        // Load existing behavior profiles
        await this.loadBehaviorProfiles();
        
        // Start background monitoring
        this.startContinuousMonitoring();
        
        console.log('API abuse prevention system initialized');
    }
    
    async analyzeRequest(requestData) {
        try {
            const clientId = requestData.clientId;
            const endpoint = requestData.endpoint;
            const timestamp = Date.now();
            
            // Update client behavior profile
            await this.updateBehaviorProfile(clientId, requestData);
            
            // Analyze for suspicious patterns
            const suspicionScore = await this.calculateSuspicionScore(clientId, requestData);
            
            // Check against attack patterns
            const threatAssessment = await this.assessThreatLevel(clientId, requestData, suspicionScore);
            
            // Apply mitigation if necessary
            if (threatAssessment.threat_level > 0.5) {
                const mitigation = await this.applyMitigation(threatAssessment);
                return {
                    allowed: mitigation.allow_request,
                    mitigation: mitigation.actions,
                    threat_level: threatAssessment.threat_level,
                    reason: threatAssessment.reason
                };
            }
            
            return {
                allowed: true,
                threat_level: suspicionScore,
                reason: 'normal_usage'
            };
            
        } catch (error) {
            console.error('Request analysis error:', error);
            return { allowed: true, error: 'analysis_failed' };
        }
    }
    
    async updateBehaviorProfile(clientId, requestData) {
        const profileKey = `behavior_profile:${clientId}`;
        const currentHour = Math.floor(Date.now() / 3600000);
        
        // Get or create behavior profile
        let profile = this.clientBehaviorProfiles.get(clientId);
        if (!profile) {
            profile = {
                client_id: clientId,
                creation_time: Date.now(),
                request_patterns: {},
                endpoint_usage: {},
                temporal_patterns: {},
                content_patterns: {},
                anomaly_scores: []
            };
        }
        
        // Update request patterns
        const requestPattern = `${requestData.method}:${requestData.endpoint}`;
        profile.request_patterns[requestPattern] = (profile.request_patterns[requestPattern] || 0) + 1;
        
        // Update endpoint usage
        profile.endpoint_usage[requestData.endpoint] = (profile.endpoint_usage[requestData.endpoint] || 0) + 1;
        
        // Update temporal patterns
        profile.temporal_patterns[currentHour] = (profile.temporal_patterns[currentHour] || 0) + 1;
        
        // Update content patterns if available
        if (requestData.contentHash) {
            profile.content_patterns[requestData.contentHash] = (profile.content_patterns[requestData.contentHash] || 0) + 1;
        }
        
        // Store updated profile
        this.clientBehaviorProfiles.set(clientId, profile);
        await this.redis.setex(profileKey, 86400 * 7, JSON.stringify(profile)); // 7 days
    }
    
    async calculateSuspicionScore(clientId, requestData) {
        const profile = this.clientBehaviorProfiles.get(clientId);
        if (!profile) return 0.1; // New clients start with low suspicion
        
        let suspicionScore = 0;
        
        // Analyze request frequency patterns
        const requestFrequency = await this.analyzeRequestFrequency(clientId);
        if (requestFrequency.anomaly_score > 0.7) {
            suspicionScore += 0.3;
        }
        
        // Analyze endpoint access patterns
        const endpointPattern = this.analyzeEndpointPattern(profile.endpoint_usage);
        if (endpointPattern.unusual_access) {
            suspicionScore += 0.2;
        }
        
        // Analyze temporal patterns
        const temporalAnomaly = this.analyzeTemporalPattern(profile.temporal_patterns);
        if (temporalAnomaly.score > 0.6) {
            suspicionScore += 0.2;
        }
        
        // Analyze content diversity
        const contentDiversity = this.analyzeContentDiversity(profile.content_patterns);
        if (contentDiversity.diversity_score < 0.3) {
            suspicionScore += 0.15;
        }
        
        // Check for distributed attack patterns
        const distributedPattern = await this.checkDistributedAttackPattern(requestData);
        if (distributedPattern.detected) {
            suspicionScore += 0.25;
        }
        
        return Math.min(1, suspicionScore);
    }
    
    async analyzeRequestFrequency(clientId) {
        const now = Date.now();
        const timeWindows = [60, 300, 3600]; // 1 min, 5 min, 1 hour
        
        let maxAnomalyScore = 0;
        
        for (const window of timeWindows) {
            const windowStart = now - (window * 1000);
            const requestCount = await this.redis.zcount(
                `requests:${clientId}`, 
                windowStart, 
                now
            );
            
            // Calculate expected requests based on historical data
            const expectedRequests = await this.getExpectedRequestCount(clientId, window);
            
            // Calculate anomaly score
            let anomalyScore = 0;
            if (expectedRequests > 0) {
                const ratio = requestCount / expectedRequests;
                if (ratio > 2) {
                    anomalyScore = Math.min(1, (ratio - 2) / 3); // Scale 2x-5x to 0-1
                }
            } else if (requestCount > 100) {
                anomalyScore = 0.8; // High suspicion for new clients with high volume
            }
            
            maxAnomalyScore = Math.max(maxAnomalyScore, anomalyScore);
        }
        
        return {
            anomaly_score: maxAnomalyScore,
            analysis_timestamp: now
        };
    }
    
    analyzeEndpointPattern(endpointUsage) {
        const endpoints = Object.keys(endpointUsage);
        const totalRequests = Object.values(endpointUsage).reduce((a, b) => a + b, 0);
        
        // Check for unusual endpoint access patterns
        let unusualAccess = false;
        let suspiciousPatterns = [];
        
        // Pattern 1: Validation-only usage (potential scraping)
        const validationRequests = endpointUsage['/api/validate'] || 0;
        if (validationRequests / totalRequests > 0.9 && totalRequests > 1000) {
            unusualAccess = true;
            suspiciousPatterns.push('validation_heavy_usage');
        }
        
        // Pattern 2: Sequential endpoint scanning
        const sequentialPatterns = this.detectSequentialScanning(endpoints);
        if (sequentialPatterns.detected) {
            unusualAccess = true;
            suspiciousPatterns.push('sequential_scanning');
        }
        
        // Pattern 3: Unused legitimate endpoints
        const legitEndpoints = ['/api/send', '/api/campaigns', '/api/templates'];
        const legitUsage = legitEndpoints.reduce((sum, ep) => sum + (endpointUsage[ep] || 0), 0);
        if (legitUsage / totalRequests < 0.1 && totalRequests > 500) {
            unusualAccess = true;
            suspiciousPatterns.push('avoiding_legitimate_endpoints');
        }
        
        return {
            unusual_access: unusualAccess,
            patterns: suspiciousPatterns,
            endpoint_diversity: endpoints.length / Math.max(1, Math.log10(totalRequests))
        };
    }
    
    detectSequentialScanning(endpoints) {
        // Look for patterns that suggest automated scanning
        const numericEndpoints = endpoints.filter(ep => /\/\d+(?:\/|$)/.test(ep));
        const sequentialIds = numericEndpoints
            .map(ep => parseInt(ep.match(/\/(\d+)(?:\/|$)/)?.[1]))
            .filter(id => !isNaN(id))
            .sort((a, b) => a - b);
        
        if (sequentialIds.length < 10) return { detected: false };
        
        // Check for consecutive sequences
        let consecutiveCount = 1;
        let maxConsecutive = 1;
        
        for (let i = 1; i < sequentialIds.length; i++) {
            if (sequentialIds[i] - sequentialIds[i-1] === 1) {
                consecutiveCount++;
            } else {
                maxConsecutive = Math.max(maxConsecutive, consecutiveCount);
                consecutiveCount = 1;
            }
        }
        maxConsecutive = Math.max(maxConsecutive, consecutiveCount);
        
        return {
            detected: maxConsecutive >= 10,
            max_consecutive: maxConsecutive,
            total_numeric_endpoints: sequentialIds.length
        };
    }
    
    analyzeTemporalPattern(temporalPatterns) {
        const hours = Object.keys(temporalPatterns).map(Number);
        const requestCounts = Object.values(temporalPatterns);
        const totalRequests = requestCounts.reduce((a, b) => a + b, 0);
        
        if (totalRequests === 0) return { score: 0 };
        
        // Calculate entropy of temporal distribution
        let entropy = 0;
        for (const count of requestCounts) {
            if (count > 0) {
                const probability = count / totalRequests;
                entropy -= probability * Math.log2(probability);
            }
        }
        
        // Normalize entropy (max entropy for 24 hours is log2(24) ≈ 4.58)
        const normalizedEntropy = entropy / 4.58;
        
        // Low entropy indicates concentrated usage (suspicious)
        // High entropy indicates distributed usage (normal)
        const anomalyScore = 1 - normalizedEntropy;
        
        // Check for 24/7 automated patterns
        const activeHours = requestCounts.filter(count => count > 0).length;
        let automatedPattern = false;
        if (activeHours >= 20 && totalRequests > 1000) {
            // Check for suspiciously regular intervals
            const avgRequestsPerActiveHour = totalRequests / activeHours;
            const variance = requestCounts.reduce((sum, count) => {
                return sum + Math.pow(count - avgRequestsPerActiveHour, 2);
            }, 0) / activeHours;
            
            const coefficientOfVariation = Math.sqrt(variance) / avgRequestsPerActiveHour;
            if (coefficientOfVariation < 0.2) {
                automatedPattern = true;
            }
        }
        
        return {
            score: automatedPattern ? Math.max(anomalyScore, 0.8) : anomalyScore,
            entropy: normalizedEntropy,
            active_hours: activeHours,
            automated_pattern: automatedPattern
        };
    }
    
    analyzeContentDiversity(contentPatterns) {
        const contentHashes = Object.keys(contentPatterns);
        const usageCounts = Object.values(contentPatterns);
        const totalUsage = usageCounts.reduce((a, b) => a + b, 0);
        
        if (totalUsage === 0) return { diversity_score: 1 };
        
        // Calculate content diversity using Shannon entropy
        let entropy = 0;
        for (const count of usageCounts) {
            const probability = count / totalUsage;
            entropy -= probability * Math.log2(probability);
        }
        
        // Normalize by maximum possible entropy
        const maxEntropy = Math.log2(contentHashes.length);
        const diversityScore = maxEntropy > 0 ? entropy / maxEntropy : 1;
        
        // Check for duplicate content abuse
        const maxReuse = Math.max(...usageCounts);
        const reuseRatio = maxReuse / totalUsage;
        
        return {
            diversity_score: diversityScore,
            unique_content_count: contentHashes.length,
            total_usage: totalUsage,
            max_reuse_ratio: reuseRatio,
            suspected_template_abuse: reuseRatio > 0.8 && totalUsage > 100
        };
    }
    
    async checkDistributedAttackPattern(requestData) {
        const timeWindow = 300000; // 5 minutes
        const now = Date.now();
        const windowStart = now - timeWindow;
        
        // Check for similar requests from different IPs/clients
        const similarRequestKey = this.generateRequestSignature(requestData);
        const similarRequestsKey = `similar_requests:${similarRequestKey}`;
        
        // Add current request
        await this.redis.zadd(similarRequestsKey, now, `${requestData.clientId}:${requestData.ip}`);
        await this.redis.expire(similarRequestsKey, 600); // 10 minutes
        
        // Count unique clients making similar requests
        const similarClients = await this.redis.zrangebyscore(
            similarRequestsKey, 
            windowStart, 
            now
        );
        
        const uniqueClients = new Set(similarClients.map(entry => entry.split(':')[0]));
        const uniqueIPs = new Set(similarClients.map(entry => entry.split(':')[1]));
        
        // Detect distributed attack patterns
        const isDistributed = uniqueClients.size >= 5 && uniqueIPs.size >= 3 && similarClients.length >= 20;
        
        return {
            detected: isDistributed,
            unique_clients: uniqueClients.size,
            unique_ips: uniqueIPs.size,
            total_similar_requests: similarClients.length,
            time_window: timeWindow / 1000
        };
    }
    
    generateRequestSignature(requestData) {
        // Create a signature for similar requests
        const signature = `${requestData.method}:${requestData.endpoint}:${requestData.contentHash || 'no-content'}`;
        return Buffer.from(signature).toString('base64').substring(0, 16);
    }
    
    async assessThreatLevel(clientId, requestData, suspicionScore) {
        let threatLevel = suspicionScore;
        let detectedPatterns = [];
        let reason = 'elevated_suspicion';
        
        // Check against known attack patterns
        for (const [patternName, pattern] of Object.entries(this.attackPatterns)) {
            const patternScore = await this.evaluateAttackPattern(clientId, requestData, pattern);
            
            if (patternScore >= pattern.threshold_score) {
                threatLevel = Math.max(threatLevel, patternScore);
                detectedPatterns.push(patternName);
                reason = `attack_pattern_detected:${patternName}`;
            }
        }
        
        return {
            threat_level: threatLevel,
            detected_patterns: detectedPatterns,
            reason: reason,
            timestamp: Date.now()
        };
    }
    
    async evaluateAttackPattern(clientId, requestData, pattern) {
        let patternScore = 0;
        const profile = this.clientBehaviorProfiles.get(clientId);
        
        for (const indicator of pattern.indicators) {
            const indicatorScore = await this.evaluateIndicator(indicator, clientId, requestData, profile);
            patternScore = Math.max(patternScore, indicatorScore);
        }
        
        return patternScore;
    }
    
    async evaluateIndicator(indicator, clientId, requestData, profile) {
        switch (indicator) {
            case 'high_failure_rate':
                return await this.checkHighFailureRate(clientId);
                
            case 'distributed_ips':
                return await this.checkDistributedIPs(clientId);
                
            case 'bulk_sending_spikes':
                return await this.checkBulkSendingSpikes(clientId, requestData);
                
            case 'validation_only_usage':
                return this.checkValidationOnlyUsage(profile);
                
            case 'sequential_scanning':
                return this.checkSequentialScanning(profile);
                
            case 'rapid_requests':
                return await this.checkRapidRequests(clientId);
                
            case 'automated_patterns':
                return this.checkAutomatedPatterns(profile);
                
            default:
                return 0;
        }
    }
    
    async checkHighFailureRate(clientId) {
        const timeWindow = 3600000; // 1 hour
        const now = Date.now();
        const windowStart = now - timeWindow;
        
        const totalRequests = await this.redis.zcount(`requests:${clientId}`, windowStart, now);
        const failedRequests = await this.redis.zcount(`failed_requests:${clientId}`, windowStart, now);
        
        if (totalRequests === 0) return 0;
        
        const failureRate = failedRequests / totalRequests;
        return failureRate > 0.5 ? Math.min(1, failureRate * 1.5) : 0;
    }
    
    async applyMitigation(threatAssessment) {
        const mitigationActions = [];
        let allowRequest = true;
        
        if (threatAssessment.threat_level >= 0.9) {
            // Critical threat - block immediately
            allowRequest = false;
            mitigationActions.push('immediate_block');
            
        } else if (threatAssessment.threat_level >= 0.7) {
            // High threat - apply strong restrictions
            mitigationActions.push('rate_limit_severe');
            mitigationActions.push('require_additional_auth');
            
        } else if (threatAssessment.threat_level >= 0.5) {
            // Medium threat - apply moderate restrictions
            mitigationActions.push('rate_limit_moderate');
            mitigationActions.push('increase_monitoring');
        }
        
        // Apply specific mitigations based on detected patterns
        for (const pattern of threatAssessment.detected_patterns) {
            const patternMitigation = this.attackPatterns[pattern].mitigation;
            if (!mitigationActions.includes(patternMitigation)) {
                mitigationActions.push(patternMitigation);
            }
        }
        
        return {
            allow_request: allowRequest,
            actions: mitigationActions,
            threat_level: threatAssessment.threat_level,
            expires_at: Date.now() + (3600000) // 1 hour
        };
    }
    
    startContinuousMonitoring() {
        // Monitor for emerging attack patterns
        setInterval(() => {
            this.analyzeGlobalPatterns();
        }, 300000); // Every 5 minutes
        
        // Clean up old data
        setInterval(() => {
            this.cleanupOldData();
        }, 3600000); // Every hour
        
        console.log('Continuous monitoring started');
    }
    
    async analyzeGlobalPatterns() {
        try {
            // Analyze patterns across all clients
            const globalPatterns = await this.detectGlobalAttackPatterns();
            
            if (globalPatterns.coordinated_attack_detected) {
                await this.alertManager.sendAlert({
                    type: 'coordinated_attack',
                    severity: 'high',
                    details: globalPatterns
                });
            }
            
        } catch (error) {
            console.error('Global pattern analysis error:', error);
        }
    }
    
    async detectGlobalAttackPatterns() {
        // This would implement advanced analytics to detect
        // coordinated attacks across multiple clients/IPs
        return {
            coordinated_attack_detected: false,
            attack_vector: null,
            affected_clients: [],
            mitigation_recommended: []
        };
    }
}

// Usage example with Express.js middleware
const express = require('express');
const app = express();

const abusePreventionConfig = {
    redis: {
        host: 'localhost',
        port: 6379
    },
    alerting: {
        webhook_url: 'https://alerts.company.com/webhook',
        email_recipients: ['security@company.com']
    }
};

const abusePreventionSystem = new EmailAPIAbusePreventionSystem(abusePreventionConfig);

// Middleware to check for abuse
app.use(async (req, res, next) => {
    try {
        const requestData = {
            clientId: req.headers['x-client-id'],
            ip: req.ip,
            endpoint: req.path,
            method: req.method,
            contentHash: req.headers['x-content-hash'],
            timestamp: Date.now()
        };
        
        const analysisResult = await abusePreventionSystem.analyzeRequest(requestData);
        
        if (!analysisResult.allowed) {
            return res.status(429).json({
                error: 'Request blocked due to suspicious activity',
                threat_level: analysisResult.threat_level,
                reason: analysisResult.reason
            });
        }
        
        // Add threat info to request for logging
        req.threatInfo = {
            threat_level: analysisResult.threat_level,
            mitigation: analysisResult.mitigation
        };
        
        next();
        
    } catch (error) {
        console.error('Abuse prevention middleware error:', error);
        next(); // Allow request through if analysis fails
    }
});
```
{% endraw %}

## Compliance and Audit Infrastructure

### Comprehensive Audit Logging

Implement detailed audit trails for security compliance:

**Request Audit Framework:**
- Complete request and response logging with correlation IDs
- Authentication attempt tracking with success/failure analysis
- Authorization decision logging with policy evaluation details
- Data access tracking for privacy compliance requirements

**Security Event Documentation:**
- Real-time security event logging with threat level classification
- Automated evidence collection for security incident investigation
- Compliance report generation for regulatory requirements
- Long-term audit data retention with encryption and integrity verification

**Performance and Health Monitoring:**
- API performance metrics with security correlation analysis
- System health indicators for security infrastructure components
- Capacity planning data for security system scaling
- Alerting integration for proactive security incident response

## Conclusion

Email API security represents a critical foundation for protecting sensitive email infrastructure and maintaining customer trust in an increasingly complex threat landscape. Organizations implementing comprehensive API security frameworks achieve significant reductions in security incidents while maintaining optimal performance for legitimate users.

Success in API security requires sophisticated authentication mechanisms, advanced rate limiting strategies, and proactive abuse prevention systems that adapt to evolving attack patterns. By following these implementation frameworks and maintaining focus on both security and usability, development teams can build resilient email APIs that protect against threats while supporting business growth.

The investment in robust API security infrastructure pays dividends through reduced security incidents, improved compliance posture, and enhanced customer confidence. In today's interconnected email ecosystem, API security capabilities often determine the difference between successful email operations and those compromised by security vulnerabilities.

Remember that API security is an ongoing discipline requiring continuous monitoring, regular security assessments, and rapid response to emerging threats. Combining advanced security measures with [professional email verification services](/services/) ensures comprehensive protection while maintaining the performance and reliability required for mission-critical email operations.