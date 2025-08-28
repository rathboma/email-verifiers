---
layout: post
title: "Email Authentication & Security Implementation: Complete Guide for SPF, DKIM, and DMARC"
date: 2025-08-27 08:00:00 -0500
categories: authentication security deliverability technical
excerpt: "Master email authentication protocols with comprehensive implementation guides for SPF, DKIM, and DMARC. Learn advanced security configurations, monitoring strategies, and troubleshooting techniques that protect your domain and ensure inbox delivery."
---

# Email Authentication & Security Implementation: Complete Guide for SPF, DKIM, and DMARC

Email authentication has become critical for successful email delivery and brand protection. With increasing sophistication of phishing attacks and stricter inbox provider policies, implementing robust SPF, DKIM, and DMARC protocols is no longer optional for serious email senders.

This comprehensive guide provides technical implementation details, advanced configuration strategies, and monitoring frameworks needed to establish bulletproof email authentication that maximizes deliverability while protecting your domain from abuse.

## Understanding Email Authentication Fundamentals

Email authentication protocols work together to verify sender identity and prevent email spoofing:

### The Authentication Triangle
- **SPF (Sender Policy Framework)**: Authorizes sending IP addresses
- **DKIM (DomainKeys Identified Mail)**: Cryptographically signs email content
- **DMARC (Domain-based Message Authentication, Reporting & Conformance)**: Provides policy enforcement and reporting

### Why Authentication Matters
- **Deliverability**: Major ISPs require authentication for inbox placement
- **Brand Protection**: Prevents domain spoofing and phishing attacks
- **Reputation**: Builds sender credibility with email providers
- **Compliance**: Meets enterprise security and regulatory requirements

## SPF Implementation and Optimization

### 1. SPF Record Structure

SPF records define which IP addresses are authorized to send email for your domain:

```
v=spf1 include:_spf.google.com include:mailgun.org ip4:192.168.1.100 -all
```

**Key Components:**
- `v=spf1`: SPF version identifier
- `include:`: References to other domains' SPF records
- `ip4:` / `ip6:`: Specific IP addresses authorized to send
- `a` / `mx`: Authorizes A or MX record IPs
- `all`: Policy for non-matching sources (`-all`, `~all`, `?all`, `+all`)

### 2. Advanced SPF Configuration

```python
# SPF record generation and validation tool
import dns.resolver
import ipaddress
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

@dataclass
class SPFMechanism:
    mechanism_type: str
    qualifier: str = "+"  # +, -, ~, ?
    value: Optional[str] = None

@dataclass
class SPFRecord:
    domain: str
    mechanisms: List[SPFMechanism]
    redirect: Optional[str] = None
    explanation: Optional[str] = None

class SPFBuilder:
    def __init__(self):
        self.authorized_ips = set()
        self.includes = []
        self.mechanisms = []
        
    def add_ip_range(self, ip_range: str):
        """Add IP address or CIDR range to authorized senders"""
        try:
            network = ipaddress.ip_network(ip_range, strict=False)
            if network.version == 4:
                self.mechanisms.append(SPFMechanism("ip4", "+", str(network)))
            else:
                self.mechanisms.append(SPFMechanism("ip6", "+", str(network)))
        except ValueError as e:
            raise ValueError(f"Invalid IP range {ip_range}: {str(e)}")
    
    def add_include(self, domain: str):
        """Add include mechanism for third-party services"""
        if not self._validate_domain(domain):
            raise ValueError(f"Invalid domain: {domain}")
        self.mechanisms.append(SPFMechanism("include", "+", domain))
    
    def add_mx_record(self, domain: Optional[str] = None):
        """Authorize MX record IPs"""
        if domain:
            self.mechanisms.append(SPFMechanism("mx", "+", domain))
        else:
            self.mechanisms.append(SPFMechanism("mx", "+"))
    
    def add_a_record(self, domain: Optional[str] = None):
        """Authorize A record IPs"""
        if domain:
            self.mechanisms.append(SPFMechanism("a", "+", domain))
        else:
            self.mechanisms.append(SPFMechanism("a", "+"))
    
    def set_default_policy(self, policy: str = "-all"):
        """Set default policy for non-matching sources"""
        valid_policies = ["-all", "~all", "?all", "+all"]
        if policy not in valid_policies:
            raise ValueError(f"Invalid policy {policy}. Must be one of: {valid_policies}")
        self.mechanisms.append(SPFMechanism("all", policy[0]))
    
    def build(self) -> str:
        """Generate SPF record string"""
        if not self.mechanisms:
            raise ValueError("No mechanisms defined")
        
        record_parts = ["v=spf1"]
        
        for mechanism in self.mechanisms:
            if mechanism.value:
                if mechanism.qualifier == "+":
                    record_parts.append(f"{mechanism.mechanism_type}:{mechanism.value}")
                else:
                    record_parts.append(f"{mechanism.qualifier}{mechanism.mechanism_type}:{mechanism.value}")
            else:
                if mechanism.qualifier == "+":
                    record_parts.append(mechanism.mechanism_type)
                else:
                    record_parts.append(f"{mechanism.qualifier}{mechanism.mechanism_type}")
        
        record = " ".join(record_parts)
        
        # Validate record length (SPF records must be under 255 characters)
        if len(record) > 255:
            raise ValueError(f"SPF record too long ({len(record)} chars). Maximum is 255 characters.")
        
        return record
    
    def _validate_domain(self, domain: str) -> bool:
        """Validate domain name format"""
        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        )
        return bool(domain_pattern.match(domain))

class SPFValidator:
    def __init__(self):
        self.dns_cache = {}
    
    def validate_record(self, domain: str) -> Dict[str, any]:
        """Validate SPF record for domain"""
        try:
            # Get SPF record
            spf_record = self._get_spf_record(domain)
            if not spf_record:
                return {
                    'valid': False,
                    'error': 'No SPF record found',
                    'recommendations': ['Add SPF record to DNS']
                }
            
            # Parse record
            mechanisms = self._parse_spf_record(spf_record)
            
            # Validate mechanisms
            validation_results = {
                'valid': True,
                'record': spf_record,
                'mechanisms': mechanisms,
                'warnings': [],
                'errors': [],
                'recommendations': []
            }
            
            # Check for common issues
            self._check_dns_lookups(mechanisms, validation_results)
            self._check_record_length(spf_record, validation_results)
            self._check_policy_strength(mechanisms, validation_results)
            self._check_includes(mechanisms, validation_results)
            
            return validation_results
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'recommendations': ['Check DNS configuration and record syntax']
            }
    
    def _get_spf_record(self, domain: str) -> Optional[str]:
        """Retrieve SPF record from DNS"""
        try:
            txt_records = dns.resolver.resolve(domain, 'TXT')
            for record in txt_records:
                record_text = str(record).strip('"')
                if record_text.startswith('v=spf1'):
                    return record_text
            return None
        except:
            return None
    
    def _parse_spf_record(self, record: str) -> List[Dict]:
        """Parse SPF record into mechanisms"""
        parts = record.split()[1:]  # Skip v=spf1
        mechanisms = []
        
        for part in parts:
            if part.startswith(('include:', 'a:', 'mx:', 'ip4:', 'ip6:', 'exists:')):
                mechanism_type, value = part.split(':', 1)
                mechanisms.append({
                    'type': mechanism_type,
                    'qualifier': '+',
                    'value': value
                })
            elif part in ['a', 'mx']:
                mechanisms.append({
                    'type': part,
                    'qualifier': '+',
                    'value': None
                })
            elif part in ['+all', '-all', '~all', '?all']:
                mechanisms.append({
                    'type': 'all',
                    'qualifier': part[0],
                    'value': None
                })
            elif part.startswith(('~', '-', '?', '+')):
                qualifier = part[0]
                mechanism = part[1:]
                if ':' in mechanism:
                    mech_type, value = mechanism.split(':', 1)
                    mechanisms.append({
                        'type': mech_type,
                        'qualifier': qualifier,
                        'value': value
                    })
                else:
                    mechanisms.append({
                        'type': mechanism,
                        'qualifier': qualifier,
                        'value': None
                    })
        
        return mechanisms
    
    def _check_dns_lookups(self, mechanisms: List[Dict], results: Dict):
        """Check if DNS lookup count exceeds limit"""
        lookup_count = 0
        
        for mechanism in mechanisms:
            if mechanism['type'] in ['include', 'a', 'mx', 'exists']:
                lookup_count += 1
            elif mechanism['type'] == 'redirect':
                lookup_count += 1
        
        if lookup_count > 10:
            results['errors'].append(f"Too many DNS lookups ({lookup_count}). Maximum is 10.")
            results['valid'] = False
            results['recommendations'].append("Reduce include statements or use IP ranges instead")
        elif lookup_count > 8:
            results['warnings'].append(f"High DNS lookup count ({lookup_count}). Consider optimization.")
    
    def _check_record_length(self, record: str, results: Dict):
        """Check record length"""
        if len(record) > 255:
            results['errors'].append(f"SPF record too long ({len(record)} chars). Maximum is 255.")
            results['valid'] = False
        elif len(record) > 200:
            results['warnings'].append(f"SPF record length ({len(record)} chars) is approaching limit.")
    
    def _check_policy_strength(self, mechanisms: List[Dict], results: Dict):
        """Check default policy strength"""
        all_mechanism = None
        for mechanism in mechanisms:
            if mechanism['type'] == 'all':
                all_mechanism = mechanism
                break
        
        if not all_mechanism:
            results['warnings'].append("No 'all' mechanism found. Add -all for strict policy.")
        elif all_mechanism['qualifier'] == '?':
            results['warnings'].append("Neutral policy (?all) provides no protection.")
        elif all_mechanism['qualifier'] == '+':
            results['warnings'].append("Permissive policy (+all) allows any sender.")
        elif all_mechanism['qualifier'] == '~':
            results['recommendations'].append("Consider upgrading from soft fail (~all) to hard fail (-all)")

# Example usage for different scenarios
def create_basic_spf():
    """Create basic SPF record for small business"""
    builder = SPFBuilder()
    builder.add_mx_record()  # Allow MX servers to send
    builder.add_include("_spf.google.com")  # Google Workspace
    builder.set_default_policy("-all")  # Strict policy
    return builder.build()

def create_enterprise_spf():
    """Create comprehensive SPF record for enterprise"""
    builder = SPFBuilder()
    builder.add_include("_spf.google.com")  # Google Workspace
    builder.add_include("spf.protection.outlook.com")  # Office 365
    builder.add_include("_spf.salesforce.com")  # Salesforce
    builder.add_include("servers.mcsv.net")  # Mailchimp
    builder.add_ip_range("192.168.1.0/24")  # Internal mail servers
    builder.add_ip_range("203.0.113.100")  # Specific sending server
    builder.set_default_policy("-all")
    return builder.build()

def create_high_volume_spf():
    """Create SPF for high-volume senders"""
    builder = SPFBuilder()
    # Use IP ranges instead of includes to reduce DNS lookups
    builder.add_ip_range("74.125.0.0/16")  # Google IP range
    builder.add_ip_range("40.92.0.0/15")   # Microsoft IP range
    builder.add_ip_range("192.168.100.0/24")  # Dedicated sending IPs
    builder.add_include("_spf.your-esp.com")  # Your ESP
    builder.set_default_policy("-all")
    return builder.build()

# Validation example
if __name__ == "__main__":
    # Build SPF record
    spf_record = create_enterprise_spf()
    print(f"Generated SPF: {spf_record}")
    
    # Validate existing domain
    validator = SPFValidator()
    results = validator.validate_record("example.com")
    
    print(f"Validation results:")
    print(f"Valid: {results['valid']}")
    if results.get('warnings'):
        print(f"Warnings: {results['warnings']}")
    if results.get('recommendations'):
        print(f"Recommendations: {results['recommendations']}")
```

### 3. SPF Monitoring and Troubleshooting

Monitor SPF authentication with automated checks:

```bash
#!/bin/bash
# SPF monitoring script

DOMAIN="yourdomain.com"
LOG_FILE="/var/log/spf-monitor.log"

# Check SPF record exists and is valid
check_spf_record() {
    echo "Checking SPF record for $DOMAIN..."
    
    SPF_RECORD=$(dig TXT "$DOMAIN" +short | grep "v=spf1")
    
    if [ -z "$SPF_RECORD" ]; then
        echo "ERROR: No SPF record found for $DOMAIN" | tee -a "$LOG_FILE"
        return 1
    fi
    
    echo "SPF Record: $SPF_RECORD" | tee -a "$LOG_FILE"
    
    # Check record length
    RECORD_LENGTH=${#SPF_RECORD}
    if [ $RECORD_LENGTH -gt 255 ]; then
        echo "WARNING: SPF record too long ($RECORD_LENGTH chars)" | tee -a "$LOG_FILE"
    fi
    
    return 0
}

# Test SPF validation with various IPs
test_spf_validation() {
    echo "Testing SPF validation..."
    
    # Test with authorized IP
    AUTHORIZED_TEST=$(echo "EHLO test.com
MAIL FROM: <test@$DOMAIN>
" | nc -w 5 smtp.gmail.com 25 2>/dev/null)
    
    echo "SPF test results logged" | tee -a "$LOG_FILE"
}

# Monitor SPF authentication failures
monitor_failures() {
    echo "Monitoring authentication failures..."
    
    # Parse mail logs for SPF failures (example for Postfix)
    FAILURES=$(grep -c "SPF.*fail" /var/log/mail.log 2>/dev/null || echo "0")
    
    echo "SPF failures in last check: $FAILURES" | tee -a "$LOG_FILE"
    
    if [ "$FAILURES" -gt 10 ]; then
        echo "WARNING: High SPF failure count: $FAILURES" | tee -a "$LOG_FILE"
        # Send alert (implement your alerting mechanism)
    fi
}

# Main execution
echo "$(date): Starting SPF monitoring for $DOMAIN" >> "$LOG_FILE"
check_spf_record
test_spf_validation
monitor_failures
echo "$(date): SPF monitoring complete" >> "$LOG_FILE"
```

## DKIM Implementation and Key Management

### 1. DKIM Key Generation and Configuration

```python
# DKIM key generation and management
import subprocess
import base64
import hashlib
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from typing import Tuple, Dict

class DKIMManager:
    def __init__(self, domain: str, key_size: int = 2048):
        self.domain = domain
        self.key_size = key_size
        self.keys_dir = f"/etc/dkim/{domain}"
        
    def generate_key_pair(self, selector: str) -> Tuple[str, str]:
        """Generate DKIM key pair"""
        # Ensure keys directory exists
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key for DNS
        public_der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Convert to base64 for DNS record
        public_b64 = base64.b64encode(public_der).decode('ascii')
        
        # Save private key
        private_key_path = f"{self.keys_dir}/{selector}.private"
        with open(private_key_path, 'wb') as f:
            f.write(private_pem)
        
        # Set restrictive permissions
        os.chmod(private_key_path, 0o600)
        
        return private_key_path, public_b64
    
    def create_dns_record(self, selector: str, public_key_b64: str, 
                         hash_algorithms: list = None, 
                         service_types: list = None) -> str:
        """Create DKIM DNS TXT record"""
        if hash_algorithms is None:
            hash_algorithms = ['sha256']
        
        if service_types is None:
            service_types = ['email']
        
        # Build DKIM record
        record_parts = [
            "v=DKIM1",
            f"k=rsa",
            f"h={':'.join(hash_algorithms)}",
            f"s={':'.join(service_types)}",
            f"p={public_key_b64}"
        ]
        
        return "; ".join(record_parts)
    
    def rotate_keys(self, old_selector: str, new_selector: str) -> Dict:
        """Rotate DKIM keys safely"""
        # Generate new key pair
        new_private_path, new_public_key = self.generate_key_pair(new_selector)
        
        # Create DNS record for new key
        new_dns_record = self.create_dns_record(new_selector, new_public_key)
        
        return {
            'new_selector': new_selector,
            'new_private_key': new_private_path,
            'new_dns_record': new_dns_record,
            'hostname': f"{new_selector}._domainkey.{self.domain}",
            'rotation_date': self._get_current_date(),
            'instructions': [
                f"1. Add DNS TXT record: {new_selector}._domainkey.{self.domain}",
                f"   Value: {new_dns_record}",
                "2. Wait for DNS propagation (24-48 hours)",
                "3. Update mail server configuration to use new selector",
                "4. Monitor for 7 days before removing old key",
                f"5. Remove old DNS record: {old_selector}._domainkey.{self.domain}"
            ]
        }
    
    def validate_dkim_record(self, selector: str) -> Dict:
        """Validate DKIM DNS record"""
        try:
            # Query DKIM record
            hostname = f"{selector}._domainkey.{self.domain}"
            result = subprocess.run(
                ['dig', 'TXT', hostname, '+short'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                return {
                    'valid': False,
                    'error': 'DNS query failed',
                    'hostname': hostname
                }
            
            record = result.stdout.strip().strip('"')
            
            if not record:
                return {
                    'valid': False,
                    'error': 'No DKIM record found',
                    'hostname': hostname
                }
            
            # Parse record
            record_dict = self._parse_dkim_record(record)
            
            # Validate components
            validation = {
                'valid': True,
                'hostname': hostname,
                'record': record,
                'parsed': record_dict,
                'warnings': [],
                'recommendations': []
            }
            
            self._validate_dkim_components(record_dict, validation)
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'hostname': f"{selector}._domainkey.{self.domain}"
            }
    
    def _parse_dkim_record(self, record: str) -> Dict:
        """Parse DKIM record into components"""
        components = {}
        
        # Remove quotes and semicolons, split by semicolon
        parts = record.replace('"', '').split(';')
        
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                components[key.strip()] = value.strip()
        
        return components
    
    def _validate_dkim_components(self, record_dict: Dict, validation: Dict):
        """Validate DKIM record components"""
        # Check version
        if record_dict.get('v') != 'DKIM1':
            validation['warnings'].append(f"Unexpected version: {record_dict.get('v')}")
        
        # Check key type
        if record_dict.get('k', 'rsa') != 'rsa':
            validation['warnings'].append(f"Non-RSA key type: {record_dict.get('k')}")
        
        # Check public key
        public_key = record_dict.get('p')
        if not public_key:
            validation['valid'] = False
            validation['warnings'].append("No public key (p=) found")
        elif len(public_key) < 200:
            validation['recommendations'].append("Public key appears short. Consider using 2048-bit keys.")
        
        # Check hash algorithms
        hash_algs = record_dict.get('h', 'sha256')
        if 'sha1' in hash_algs.lower():
            validation['recommendations'].append("SHA-1 is deprecated. Use SHA-256 only.")
        
        # Check flags
        flags = record_dict.get('t', '')
        if 'y' in flags:
            validation['recommendations'].append("Testing flag (t=y) should be removed in production.")
    
    def _get_current_date(self) -> str:
        """Get current date in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

# DKIM signing configuration for mail servers
def generate_postfix_dkim_config(domain: str, selector: str, private_key_path: str) -> str:
    """Generate Postfix OpenDKIM configuration"""
    config = f"""
# OpenDKIM Configuration for {domain}

# Key table (KeyTable)
{selector}._domainkey.{domain} {domain}:{selector}:{private_key_path}

# Signing table (SigningTable)
*@{domain} {selector}._domainkey.{domain}

# Trusted hosts
127.0.0.1
localhost
::1
{domain}

# OpenDKIM configuration
Canonicalization relaxed/simple
Mode sv
SubDomains no
AutoRestart yes
PidFile /var/run/opendkim/opendkim.pid
UMask 002
UserID opendkim
Socket inet:8891@localhost
"""
    return config

def generate_exim_dkim_config(domain: str, selector: str, private_key_path: str) -> str:
    """Generate Exim DKIM configuration"""
    config = f"""
# DKIM Configuration for Exim

# In the transport section:
dkim_domain = {domain}
dkim_selector = {selector}
dkim_private_key = {private_key_path}
dkim_canon = relaxed
dkim_strict = 0
dkim_sign_headers = from:to:subject:date:message-id
"""
    return config

# Usage examples
if __name__ == "__main__":
    # Initialize DKIM manager
    dkim = DKIMManager("example.com")
    
    # Generate initial key pair
    private_key_path, public_key = dkim.generate_key_pair("default")
    
    # Create DNS record
    dns_record = dkim.create_dns_record("default", public_key)
    print(f"Add this DNS TXT record:")
    print(f"Hostname: default._domainkey.example.com")
    print(f"Value: {dns_record}")
    
    # Validate existing record
    validation = dkim.validate_dkim_record("default")
    print(f"DKIM validation: {validation}")
    
    # Generate mail server configuration
    postfix_config = generate_postfix_dkim_config("example.com", "default", private_key_path)
    print(f"Postfix configuration:\n{postfix_config}")
```

### 2. DKIM Key Rotation Strategy

Implement automatic key rotation for enhanced security:

```javascript
// DKIM key rotation automation
class DKIMRotationManager {
  constructor(config) {
    this.config = config;
    this.rotationSchedule = {
      quarterly: 90,   // days
      biannual: 180,   // days
      annual: 365      // days
    };
    this.currentKeys = new Map();
    this.pendingRotations = new Map();
  }

  async scheduleKeyRotation(domain, currentSelector, rotationInterval = 'quarterly') {
    const rotationDays = this.rotationSchedule[rotationInterval];
    const nextRotationDate = new Date();
    nextRotationDate.setDate(nextRotationDate.getDate() + rotationDays);
    
    // Generate new selector based on date
    const newSelector = this.generateSelector();
    
    const rotation = {
      domain: domain,
      currentSelector: currentSelector,
      newSelector: newSelector,
      scheduledDate: nextRotationDate,
      status: 'scheduled',
      phases: [
        { name: 'generate', completed: false, date: null },
        { name: 'dns_publish', completed: false, date: null },
        { name: 'propagation_wait', completed: false, date: null },
        { name: 'mail_server_update', completed: false, date: null },
        { name: 'monitoring', completed: false, date: null },
        { name: 'old_key_removal', completed: false, date: null }
      ]
    };
    
    this.pendingRotations.set(`${domain}-${newSelector}`, rotation);
    
    // Schedule execution
    setTimeout(() => {
      this.executeKeyRotation(domain, rotation);
    }, this.calculateDelay(nextRotationDate));
    
    return rotation;
  }

  async executeKeyRotation(domain, rotation) {
    console.log(`Starting DKIM key rotation for ${domain}`);
    
    try {
      // Phase 1: Generate new key pair
      await this.phaseGenerateKeys(domain, rotation);
      
      // Phase 2: Publish DNS record
      await this.phasePublishDNS(domain, rotation);
      
      // Phase 3: Wait for DNS propagation
      await this.phaseDNSPropagation(domain, rotation);
      
      // Phase 4: Update mail server configuration
      await this.phaseUpdateMailServer(domain, rotation);
      
      // Phase 5: Monitor for issues
      await this.phaseMonitoring(domain, rotation);
      
      // Phase 6: Remove old key
      await this.phaseRemoveOldKey(domain, rotation);
      
      console.log(`DKIM key rotation completed for ${domain}`);
      
    } catch (error) {
      console.error(`DKIM rotation failed for ${domain}:`, error);
      await this.handleRotationFailure(domain, rotation, error);
    }
  }

  async phaseGenerateKeys(domain, rotation) {
    console.log(`Phase 1: Generating new DKIM keys for ${domain}`);
    
    // Call DKIM manager to generate new key pair
    const response = await fetch('/api/dkim/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        domain: domain,
        selector: rotation.newSelector,
        keySize: this.config.keySize || 2048
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to generate keys: ${response.statusText}`);
    }
    
    const keyData = await response.json();
    rotation.newPrivateKeyPath = keyData.privateKeyPath;
    rotation.newPublicKey = keyData.publicKey;
    rotation.newDNSRecord = keyData.dnsRecord;
    
    this.completePhase(rotation, 'generate');
  }

  async phasePublishDNS(domain, rotation) {
    console.log(`Phase 2: Publishing DNS record for ${domain}`);
    
    // Publish new DKIM DNS record
    const hostname = `${rotation.newSelector}._domainkey.${domain}`;
    
    const response = await fetch('/api/dns/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hostname: hostname,
        type: 'TXT',
        value: rotation.newDNSRecord,
        ttl: 300  // Short TTL for faster propagation
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to publish DNS: ${response.statusText}`);
    }
    
    this.completePhase(rotation, 'dns_publish');
  }

  async phaseDNSPropagation(domain, rotation) {
    console.log(`Phase 3: Waiting for DNS propagation for ${domain}`);
    
    const hostname = `${rotation.newSelector}._domainkey.${domain}`;
    const maxWaitTime = 48 * 60 * 60 * 1000; // 48 hours
    const checkInterval = 5 * 60 * 1000; // 5 minutes
    
    let waitTime = 0;
    let propagated = false;
    
    while (waitTime < maxWaitTime && !propagated) {
      await this.sleep(checkInterval);
      waitTime += checkInterval;
      
      // Check DNS propagation from multiple resolvers
      const propagationResults = await this.checkDNSPropagation(hostname);
      
      if (propagationResults.propagated) {
        propagated = true;
        console.log(`DNS propagated after ${waitTime / (60 * 1000)} minutes`);
      } else {
        console.log(`DNS propagation: ${propagationResults.successCount}/${propagationResults.totalChecks} resolvers`);
      }
    }
    
    if (!propagated) {
      throw new Error(`DNS propagation timeout after ${maxWaitTime / (60 * 60 * 1000)} hours`);
    }
    
    this.completePhase(rotation, 'propagation_wait');
  }

  async phaseUpdateMailServer(domain, rotation) {
    console.log(`Phase 4: Updating mail server configuration for ${domain}`);
    
    // Update mail server configuration
    const response = await fetch('/api/mailserver/update-dkim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        domain: domain,
        newSelector: rotation.newSelector,
        privateKeyPath: rotation.newPrivateKeyPath,
        gracefulTransition: true
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update mail server: ${response.statusText}`);
    }
    
    // Test DKIM signing with new key
    const signingTest = await this.testDKIMSigning(domain, rotation.newSelector);
    if (!signingTest.success) {
      throw new Error(`DKIM signing test failed: ${signingTest.error}`);
    }
    
    this.completePhase(rotation, 'mail_server_update');
  }

  async phaseMonitoring(domain, rotation) {
    console.log(`Phase 5: Monitoring DKIM performance for ${domain}`);
    
    const monitoringPeriod = 7 * 24 * 60 * 60 * 1000; // 7 days
    const checkInterval = 60 * 60 * 1000; // 1 hour
    
    let monitoringTime = 0;
    const issues = [];
    
    while (monitoringTime < monitoringPeriod) {
      await this.sleep(checkInterval);
      monitoringTime += checkInterval;
      
      // Check DKIM authentication rates
      const authStats = await this.getDKIMAuthenticationStats(domain, rotation.newSelector);
      
      if (authStats.failureRate > 0.05) { // 5% failure threshold
        issues.push(`High DKIM failure rate: ${(authStats.failureRate * 100).toFixed(2)}%`);
      }
      
      // Check for delivery issues
      const deliveryStats = await this.getDeliveryStats(domain);
      if (deliveryStats.bounceRate > 0.02) { // 2% bounce threshold
        issues.push(`High bounce rate: ${(deliveryStats.bounceRate * 100).toFixed(2)}%`);
      }
      
      console.log(`Monitoring progress: ${((monitoringTime / monitoringPeriod) * 100).toFixed(1)}% complete`);
    }
    
    if (issues.length > 0) {
      console.warn(`Issues detected during monitoring: ${issues.join(', ')}`);
      // Continue but log issues for investigation
    }
    
    this.completePhase(rotation, 'monitoring');
  }

  async phaseRemoveOldKey(domain, rotation) {
    console.log(`Phase 6: Removing old DKIM key for ${domain}`);
    
    // Remove old DNS record
    const oldHostname = `${rotation.currentSelector}._domainkey.${domain}`;
    
    const dnsResponse = await fetch('/api/dns/delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hostname: oldHostname,
        type: 'TXT'
      })
    });
    
    if (!dnsResponse.ok) {
      console.warn(`Failed to remove old DNS record: ${dnsResponse.statusText}`);
    }
    
    // Remove old private key file (optional, for security)
    const keyResponse = await fetch('/api/dkim/cleanup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        domain: domain,
        selector: rotation.currentSelector
      })
    });
    
    if (!keyResponse.ok) {
      console.warn(`Failed to cleanup old key files: ${keyResponse.statusText}`);
    }
    
    this.completePhase(rotation, 'old_key_removal');
    
    // Update current key tracking
    this.currentKeys.set(domain, rotation.newSelector);
  }

  async checkDNSPropagation(hostname) {
    const publicResolvers = [
      '8.8.8.8',      // Google
      '1.1.1.1',      // Cloudflare  
      '208.67.222.222', // OpenDNS
      '64.6.64.6'     // Verisign
    ];
    
    let successCount = 0;
    
    for (const resolver of publicResolvers) {
      try {
        const result = await this.queryDNS(hostname, 'TXT', resolver);
        if (result && result.includes('v=DKIM1')) {
          successCount++;
        }
      } catch (error) {
        console.log(`DNS check failed for resolver ${resolver}: ${error.message}`);
      }
    }
    
    return {
      propagated: successCount === publicResolvers.length,
      successCount: successCount,
      totalChecks: publicResolvers.length
    };
  }

  generateSelector() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    return `${year}${month}`;
  }

  completePhase(rotation, phaseName) {
    const phase = rotation.phases.find(p => p.name === phaseName);
    if (phase) {
      phase.completed = true;
      phase.date = new Date();
    }
  }

  calculateDelay(targetDate) {
    return Math.max(0, targetDate.getTime() - Date.now());
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Usage example
const rotationManager = new DKIMRotationManager({
  keySize: 2048,
  monitoringPeriod: 7 // days
});

// Schedule quarterly rotation
rotationManager.scheduleKeyRotation('example.com', 'current', 'quarterly');
```

## DMARC Policy Implementation and Analysis

### 1. DMARC Policy Configuration

```python
# DMARC policy management and analysis
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import gzip
import base64

@dataclass
class DMARCReport:
    org_name: str
    email: str
    report_id: str
    date_begin: datetime
    date_end: datetime
    domain: str
    records: List[Dict]

@dataclass
class DMARCPolicy:
    version: str = "DMARC1"
    policy: str = "none"  # none, quarantine, reject
    subdomain_policy: Optional[str] = None
    percentage: int = 100
    rua_addresses: List[str] = None  # Aggregate reports
    ruf_addresses: List[str] = None  # Failure reports
    failure_options: str = "0"  # 0=all, 1=any, d=DKIM, s=SPF
    report_interval: int = 86400  # 24 hours in seconds
    alignment_spf: str = "r"  # r=relaxed, s=strict
    alignment_dkim: str = "r"  # r=relaxed, s=strict

class DMARCManager:
    def __init__(self):
        self.policies = {}
        self.reports = []
        
    def create_policy(self, domain: str, policy_level: str = "monitoring") -> str:
        """Create DMARC policy based on deployment phase"""
        
        policy_templates = {
            "monitoring": DMARCPolicy(
                policy="none",
                percentage=100,
                rua_addresses=[f"dmarc-reports@{domain}"],
                failure_options="1"
            ),
            "quarantine": DMARCPolicy(
                policy="quarantine", 
                percentage=25,  # Start with 25%
                rua_addresses=[f"dmarc-reports@{domain}"],
                ruf_addresses=[f"dmarc-failures@{domain}"],
                failure_options="1"
            ),
            "enforce": DMARCPolicy(
                policy="reject",
                percentage=100,
                subdomain_policy="reject",
                rua_addresses=[f"dmarc-reports@{domain}"],
                ruf_addresses=[f"dmarc-failures@{domain}"],
                failure_options="0"
            )
        }
        
        if policy_level not in policy_templates:
            raise ValueError(f"Invalid policy level: {policy_level}")
            
        policy = policy_templates[policy_level]
        self.policies[domain] = policy
        
        return self._build_dmarc_record(policy)
    
    def _build_dmarc_record(self, policy: DMARCPolicy) -> str:
        """Build DMARC DNS record string"""
        record_parts = [f"v={policy.version}"]
        
        # Required policy
        record_parts.append(f"p={policy.policy}")
        
        # Optional subdomain policy
        if policy.subdomain_policy:
            record_parts.append(f"sp={policy.subdomain_policy}")
        
        # Percentage
        if policy.percentage != 100:
            record_parts.append(f"pct={policy.percentage}")
        
        # Report addresses
        if policy.rua_addresses:
            rua = ",".join([f"mailto:{addr}" for addr in policy.rua_addresses])
            record_parts.append(f"rua={rua}")
            
        if policy.ruf_addresses:
            ruf = ",".join([f"mailto:{addr}" for addr in policy.ruf_addresses])
            record_parts.append(f"ruf={ruf}")
        
        # Alignment modes
        if policy.alignment_spf != "r":
            record_parts.append(f"aspf={policy.alignment_spf}")
            
        if policy.alignment_dkim != "r":
            record_parts.append(f"adkim={policy.alignment_dkim}")
        
        # Failure reporting options
        if policy.failure_options != "0":
            record_parts.append(f"fo={policy.failure_options}")
        
        # Report interval
        if policy.report_interval != 86400:
            record_parts.append(f"ri={policy.report_interval}")
        
        return "; ".join(record_parts)
    
    def parse_aggregate_report(self, report_xml: str) -> DMARCReport:
        """Parse DMARC aggregate report XML"""
        try:
            root = ET.fromstring(report_xml)
            
            # Extract metadata
            report_metadata = root.find('.//report_metadata')
            org_name = report_metadata.find('org_name').text
            email = report_metadata.find('email').text
            report_id = report_metadata.find('report_id').text
            
            date_range = report_metadata.find('date_range')
            date_begin = datetime.fromtimestamp(int(date_range.find('begin').text))
            date_end = datetime.fromtimestamp(int(date_range.find('end').text))
            
            # Extract policy published
            policy_published = root.find('.//policy_published')
            domain = policy_published.find('domain').text
            
            # Extract records
            records = []
            for record in root.findall('.//record'):
                source_ip = record.find('.//source_ip').text
                count = int(record.find('.//count').text)
                
                # Policy evaluation results
                policy_eval = record.find('.//policy_evaluated')
                disposition = policy_eval.find('disposition').text
                dkim_result = policy_eval.find('dkim').text
                spf_result = policy_eval.find('spf').text
                
                # Authentication results
                auth_results = record.find('.//auth_results')
                
                dkim_auths = []
                for dkim in auth_results.findall('.//dkim'):
                    dkim_auths.append({
                        'domain': dkim.find('domain').text,
                        'result': dkim.find('result').text,
                        'selector': dkim.find('selector').text if dkim.find('selector') is not None else None
                    })
                
                spf_auths = []
                for spf in auth_results.findall('.//spf'):
                    spf_auths.append({
                        'domain': spf.find('domain').text,
                        'result': spf.find('result').text
                    })
                
                record_data = {
                    'source_ip': source_ip,
                    'count': count,
                    'disposition': disposition,
                    'dkim_result': dkim_result,
                    'spf_result': spf_result,
                    'dkim_auth': dkim_auths,
                    'spf_auth': spf_auths
                }
                
                records.append(record_data)
            
            return DMARCReport(
                org_name=org_name,
                email=email,
                report_id=report_id,
                date_begin=date_begin,
                date_end=date_end,
                domain=domain,
                records=records
            )
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing report: {str(e)}")
    
    def analyze_reports(self, domain: str, days: int = 30) -> Dict:
        """Analyze DMARC reports for domain"""
        # Filter reports for domain and time period
        cutoff_date = datetime.now() - timedelta(days=days)
        domain_reports = [
            report for report in self.reports 
            if report.domain == domain and report.date_begin >= cutoff_date
        ]
        
        if not domain_reports:
            return {
                'domain': domain,
                'analysis_period': f"{days} days",
                'error': 'No reports found for analysis period'
            }
        
        # Aggregate statistics
        total_messages = 0
        dmarc_pass = 0
        dmarc_fail = 0
        spf_pass = 0
        spf_fail = 0
        dkim_pass = 0
        dkim_fail = 0
        
        source_ips = {}
        dispositions = {'none': 0, 'quarantine': 0, 'reject': 0}
        
        for report in domain_reports:
            for record in report.records:
                count = record['count']
                total_messages += count
                
                # Count authentication results
                if record['spf_result'] == 'pass':
                    spf_pass += count
                else:
                    spf_fail += count
                
                if record['dkim_result'] == 'pass':
                    dkim_pass += count
                else:
                    dkim_fail += count
                
                # DMARC alignment (pass if either SPF or DKIM passes)
                if record['spf_result'] == 'pass' or record['dkim_result'] == 'pass':
                    dmarc_pass += count
                else:
                    dmarc_fail += count
                
                # Track source IPs
                source_ip = record['source_ip']
                if source_ip not in source_ips:
                    source_ips[source_ip] = {'total': 0, 'pass': 0, 'fail': 0}
                
                source_ips[source_ip]['total'] += count
                if record['spf_result'] == 'pass' or record['dkim_result'] == 'pass':
                    source_ips[source_ip]['pass'] += count
                else:
                    source_ips[source_ip]['fail'] += count
                
                # Count dispositions
                disposition = record['disposition']
                if disposition in dispositions:
                    dispositions[disposition] += count
        
        # Calculate rates
        spf_pass_rate = (spf_pass / total_messages * 100) if total_messages > 0 else 0
        dkim_pass_rate = (dkim_pass / total_messages * 100) if total_messages > 0 else 0
        dmarc_pass_rate = (dmarc_pass / total_messages * 100) if total_messages > 0 else 0
        
        # Identify top failing sources
        failing_sources = []
        for ip, stats in source_ips.items():
            if stats['fail'] > 0:
                fail_rate = (stats['fail'] / stats['total']) * 100
                failing_sources.append({
                    'ip': ip,
                    'total_messages': stats['total'],
                    'failed_messages': stats['fail'],
                    'failure_rate': fail_rate
                })
        
        failing_sources.sort(key=lambda x: x['failed_messages'], reverse=True)
        
        return {
            'domain': domain,
            'analysis_period': f"{days} days",
            'total_messages': total_messages,
            'dmarc_compliance': {
                'pass_rate': dmarc_pass_rate,
                'pass_count': dmarc_pass,
                'fail_count': dmarc_fail
            },
            'spf_authentication': {
                'pass_rate': spf_pass_rate,
                'pass_count': spf_pass,
                'fail_count': spf_fail
            },
            'dkim_authentication': {
                'pass_rate': dkim_pass_rate,
                'pass_count': dkim_pass,
                'fail_count': dkim_fail
            },
            'dispositions': dispositions,
            'top_sources': len(source_ips),
            'failing_sources': failing_sources[:10],  # Top 10 failing sources
            'recommendations': self._generate_recommendations(
                dmarc_pass_rate, spf_pass_rate, dkim_pass_rate, failing_sources
            )
        }
    
    def _generate_recommendations(self, dmarc_rate: float, spf_rate: float, 
                                dkim_rate: float, failing_sources: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if dmarc_rate < 95:
            recommendations.append(
                f"DMARC compliance is {dmarc_rate:.1f}%. Investigate failing sources before enforcing policy."
            )
        
        if spf_rate < 90:
            recommendations.append(
                f"SPF pass rate is {spf_rate:.1f}%. Review SPF record and authorized sending sources."
            )
        
        if dkim_rate < 90:
            recommendations.append(
                f"DKIM pass rate is {dkim_rate:.1f}%. Verify DKIM configuration and key validity."
            )
        
        if failing_sources:
            high_volume_failures = [
                source for source in failing_sources 
                if source['total_messages'] > 100 and source['failure_rate'] > 50
            ]
            
            if high_volume_failures:
                recommendations.append(
                    f"High-volume failing sources detected: {len(high_volume_failures)} IPs. "
                    "Investigate if these are legitimate sending sources."
                )
        
        if dmarc_rate > 98 and spf_rate > 95 and dkim_rate > 95:
            recommendations.append(
                "Excellent authentication rates. Consider moving to 'quarantine' or 'reject' policy."
            )
        
        return recommendations
    
    def suggest_policy_progression(self, domain: str) -> Dict:
        """Suggest next steps for DMARC policy progression"""
        analysis = self.analyze_reports(domain)
        
        if 'error' in analysis:
            return {
                'current_recommendation': 'monitoring',
                'reason': 'Insufficient data for analysis',
                'next_steps': ['Deploy monitoring policy', 'Collect reports for 30 days']
            }
        
        dmarc_rate = analysis['dmarc_compliance']['pass_rate']
        failing_count = len(analysis['failing_sources'])
        
        if dmarc_rate >= 99 and failing_count == 0:
            return {
                'current_recommendation': 'enforce',
                'reason': 'Excellent compliance with no failing sources',
                'next_steps': [
                    'Deploy p=reject policy',
                    'Monitor for 7 days',
                    'Ensure all legitimate sources pass authentication'
                ]
            }
        elif dmarc_rate >= 95 and failing_count <= 2:
            return {
                'current_recommendation': 'quarantine',
                'reason': 'Good compliance with minimal failures',
                'next_steps': [
                    'Deploy p=quarantine policy with pct=25',
                    'Monitor for issues',
                    'Gradually increase percentage',
                    'Investigate remaining failing sources'
                ]
            }
        else:
            return {
                'current_recommendation': 'monitoring',
                'reason': f'Compliance rate {dmarc_rate:.1f}% with {failing_count} failing sources',
                'next_steps': [
                    'Continue monitoring with p=none',
                    'Fix SPF and DKIM authentication',
                    'Investigate failing IP addresses',
                    'Aim for >95% compliance before enforcement'
                ]
            }

# Usage example
if __name__ == "__main__":
    dmarc = DMARCManager()
    
    # Create monitoring policy
    monitoring_policy = dmarc.create_policy("example.com", "monitoring")
    print(f"Monitoring DMARC record: {monitoring_policy}")
    
    # Create enforcement policy  
    enforcement_policy = dmarc.create_policy("example.com", "enforce")
    print(f"Enforcement DMARC record: {enforcement_policy}")
    
    # Example report parsing (you would get this from email or API)
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feedback>
      <report_metadata>
        <org_name>google.com</org_name>
        <email>noreply-dmarc-support@google.com</email>
        <report_id>12345</report_id>
        <date_range>
          <begin>1640995200</begin>
          <end>1641081600</end>
        </date_range>
      </report_metadata>
      <policy_published>
        <domain>example.com</domain>
        <p>none</p>
        <pct>100</pct>
      </policy_published>
      <record>
        <row>
          <source_ip>192.168.1.100</source_ip>
          <count>150</count>
          <policy_evaluated>
            <disposition>none</disposition>
            <dkim>pass</dkim>
            <spf>pass</spf>
          </policy_evaluated>
        </row>
        <identifiers>
          <header_from>example.com</header_from>
        </identifiers>
        <auth_results>
          <dkim>
            <domain>example.com</domain>
            <result>pass</result>
            <selector>default</selector>
          </dkim>
          <spf>
            <domain>example.com</domain>
            <result>pass</result>
          </spf>
        </auth_results>
      </record>
    </feedback>"""
    
    # Parse and analyze report
    report = dmarc.parse_aggregate_report(sample_xml)
    dmarc.reports.append(report)
    
    analysis = dmarc.analyze_reports("example.com")
    print(f"DMARC Analysis: {json.dumps(analysis, indent=2, default=str)}")
    
    # Get policy progression recommendation
    progression = dmarc.suggest_policy_progression("example.com")
    print(f"Policy Progression: {json.dumps(progression, indent=2)}")
```

## Monitoring and Alerting Framework

### 1. Authentication Monitoring Dashboard

```python
# Authentication monitoring and alerting system
import asyncio
import aioredis
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AuthenticationMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.redis_pool = None
        self.alert_rules = {
            'spf_failure_rate': {'threshold': 5.0, 'severity': 'warning'},
            'dkim_failure_rate': {'threshold': 5.0, 'severity': 'warning'},  
            'dmarc_failure_rate': {'threshold': 10.0, 'severity': 'critical'},
            'authentication_volume_drop': {'threshold': 50.0, 'severity': 'warning'}
        }
        self.notification_channels = config.get('notification_channels', {})
        
    async def initialize(self):
        """Initialize monitoring system"""
        self.redis_pool = aioredis.ConnectionPool.from_url(
            self.config['redis_url'], 
            max_connections=10
        )
        
        # Start monitoring tasks
        asyncio.create_task(self.monitor_authentication_metrics())
        asyncio.create_task(self.process_dmarc_reports())
        
    async def track_authentication_event(self, event: Dict):
        """Track authentication event"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        
        timestamp = datetime.now()
        event_data = {
            'timestamp': timestamp.isoformat(),
            'domain': event['domain'],
            'source_ip': event.get('source_ip'),
            'spf_result': event.get('spf_result'),
            'dkim_result': event.get('dkim_result'), 
            'dmarc_result': event.get('dmarc_result'),
            'message_count': event.get('message_count', 1)
        }
        
        # Store event
        await redis.lpush('auth_events', json.dumps(event_data))
        await redis.ltrim('auth_events', 0, 100000)  # Keep last 100k events
        
        # Update real-time counters
        await self.update_real_time_counters(event_data)
        
    async def update_real_time_counters(self, event: Dict):
        """Update real-time authentication counters"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        
        domain = event['domain']
        hour_key = datetime.now().strftime('%Y-%m-%d-%H')
        
        # Increment counters
        pipeline = redis.pipeline()
        
        pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'total', event['message_count'])
        
        if event['spf_result'] == 'pass':
            pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'spf_pass', event['message_count'])
        else:
            pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'spf_fail', event['message_count'])
            
        if event['dkim_result'] == 'pass':
            pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'dkim_pass', event['message_count'])
        else:
            pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'dkim_fail', event['message_count'])
            
        if event['dmarc_result'] == 'pass':
            pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'dmarc_pass', event['message_count'])
        else:
            pipeline.hincrby(f'auth_hourly:{domain}:{hour_key}', 'dmarc_fail', event['message_count'])
        
        # Set expiration for hourly keys (keep for 30 days)
        pipeline.expire(f'auth_hourly:{domain}:{hour_key}', 30 * 24 * 3600)
        
        await pipeline.execute()
        
    async def calculate_authentication_metrics(self, domain: str, hours: int = 24) -> Dict:
        """Calculate authentication metrics for domain"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        
        metrics = {
            'domain': domain,
            'period_hours': hours,
            'total_messages': 0,
            'spf_pass': 0,
            'spf_fail': 0,
            'dkim_pass': 0,
            'dkim_fail': 0,
            'dmarc_pass': 0,
            'dmarc_fail': 0,
            'spf_pass_rate': 0.0,
            'dkim_pass_rate': 0.0,
            'dmarc_pass_rate': 0.0
        }
        
        # Get data for specified time period
        current_hour = datetime.now()
        
        for i in range(hours):
            hour_key = (current_hour - timedelta(hours=i)).strftime('%Y-%m-%d-%H')
            hour_data = await redis.hgetall(f'auth_hourly:{domain}:{hour_key}')
            
            if hour_data:
                metrics['total_messages'] += int(hour_data.get(b'total', 0))
                metrics['spf_pass'] += int(hour_data.get(b'spf_pass', 0))
                metrics['spf_fail'] += int(hour_data.get(b'spf_fail', 0))
                metrics['dkim_pass'] += int(hour_data.get(b'dkim_pass', 0))
                metrics['dkim_fail'] += int(hour_data.get(b'dkim_fail', 0))
                metrics['dmarc_pass'] += int(hour_data.get(b'dmarc_pass', 0))
                metrics['dmarc_fail'] += int(hour_data.get(b'dmarc_fail', 0))
        
        # Calculate rates
        if metrics['total_messages'] > 0:
            metrics['spf_pass_rate'] = (metrics['spf_pass'] / metrics['total_messages']) * 100
            metrics['dkim_pass_rate'] = (metrics['dkim_pass'] / metrics['total_messages']) * 100
            metrics['dmarc_pass_rate'] = (metrics['dmarc_pass'] / metrics['total_messages']) * 100
        
        return metrics
        
    async def monitor_authentication_metrics(self):
        """Main monitoring loop"""
        while True:
            try:
                # Get list of domains to monitor
                domains = await self.get_monitored_domains()
                
                for domain in domains:
                    # Calculate current metrics
                    current_metrics = await self.calculate_authentication_metrics(domain, 1)  # Last hour
                    historical_metrics = await self.calculate_authentication_metrics(domain, 24)  # Last 24 hours
                    
                    # Check for alerts
                    alerts = await self.check_alert_conditions(domain, current_metrics, historical_metrics)
                    
                    for alert in alerts:
                        await self.send_alert(alert)
                        
                    # Store metrics for trending
                    await self.store_metrics_snapshot(domain, current_metrics)
                    
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"Error in authentication monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    async def check_alert_conditions(self, domain: str, current: Dict, historical: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        # SPF failure rate alert
        if current['spf_pass_rate'] < (100 - self.alert_rules['spf_failure_rate']['threshold']):
            alerts.append({
                'type': 'spf_failure_rate',
                'domain': domain,
                'severity': self.alert_rules['spf_failure_rate']['severity'],
                'current_rate': current['spf_pass_rate'],
                'threshold': 100 - self.alert_rules['spf_failure_rate']['threshold'],
                'message': f"SPF pass rate {current['spf_pass_rate']:.1f}% below threshold"
            })
            
        # DKIM failure rate alert
        if current['dkim_pass_rate'] < (100 - self.alert_rules['dkim_failure_rate']['threshold']):
            alerts.append({
                'type': 'dkim_failure_rate',
                'domain': domain,
                'severity': self.alert_rules['dkim_failure_rate']['severity'],
                'current_rate': current['dkim_pass_rate'],
                'threshold': 100 - self.alert_rules['dkim_failure_rate']['threshold'],
                'message': f"DKIM pass rate {current['dkim_pass_rate']:.1f}% below threshold"
            })
            
        # DMARC failure rate alert
        if current['dmarc_pass_rate'] < (100 - self.alert_rules['dmarc_failure_rate']['threshold']):
            alerts.append({
                'type': 'dmarc_failure_rate',
                'domain': domain,
                'severity': self.alert_rules['dmarc_failure_rate']['severity'],
                'current_rate': current['dmarc_pass_rate'],
                'threshold': 100 - self.alert_rules['dmarc_failure_rate']['threshold'],
                'message': f"DMARC pass rate {current['dmarc_pass_rate']:.1f}% below threshold"
            })
            
        # Volume drop alert
        if (historical['total_messages'] > 100 and 
            current['total_messages'] < historical['total_messages'] * 0.5):  # 50% drop
            
            alerts.append({
                'type': 'authentication_volume_drop',
                'domain': domain,
                'severity': self.alert_rules['authentication_volume_drop']['severity'],
                'current_volume': current['total_messages'],
                'historical_volume': historical['total_messages'],
                'message': f"Authentication volume dropped from {historical['total_messages']} to {current['total_messages']}"
            })
            
        return alerts
        
    async def send_alert(self, alert: Dict):
        """Send alert through configured channels"""
        alert_message = f"""
AUTHENTICATION ALERT: {alert['domain']}

Type: {alert['type']}
Severity: {alert['severity']}
Message: {alert['message']}
Timestamp: {datetime.now().isoformat()}
"""
        
        # Send email alert
        if 'email' in self.notification_channels:
            await self.send_email_alert(alert, alert_message)
            
        # Send Slack alert
        if 'slack' in self.notification_channels:
            await self.send_slack_alert(alert, alert_message)
            
        # Send webhook alert
        if 'webhook' in self.notification_channels:
            await self.send_webhook_alert(alert)
            
    async def send_email_alert(self, alert: Dict, message: str):
        """Send email alert"""
        try:
            email_config = self.notification_channels['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{alert['severity'].upper()}] Authentication Alert - {alert['domain']}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port']) as server:
                if email_config.get('use_tls'):
                    server.starttls()
                if email_config.get('username'):
                    server.login(email_config['username'], email_config['password'])
                
                server.send_message(msg)
                
        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")
            
    async def get_monitored_domains(self) -> List[str]:
        """Get list of domains to monitor"""
        # This would typically come from configuration or database
        return self.config.get('monitored_domains', ['example.com'])
        
    async def store_metrics_snapshot(self, domain: str, metrics: Dict):
        """Store metrics snapshot for historical analysis"""
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
        snapshot_key = f'metrics_snapshot:{domain}:{timestamp}'
        
        await redis.hset(snapshot_key, mapping={
            k: str(v) for k, v in metrics.items()
        })
        
        # Set expiration (keep for 90 days)
        await redis.expire(snapshot_key, 90 * 24 * 3600)

# Usage example
async def main():
    config = {
        'redis_url': 'redis://localhost:6379',
        'monitored_domains': ['example.com', 'test.com'],
        'notification_channels': {
            'email': {
                'smtp_host': 'smtp.example.com',
                'smtp_port': 587,
                'use_tls': True,
                'username': 'alerts@example.com',
                'password': 'your_password',
                'from_address': 'alerts@example.com',
                'to_addresses': ['admin@example.com']
            }
        }
    }
    
    monitor = AuthenticationMonitor(config)
    await monitor.initialize()
    
    # Simulate authentication events
    await monitor.track_authentication_event({
        'domain': 'example.com',
        'source_ip': '192.168.1.100',
        'spf_result': 'pass',
        'dkim_result': 'pass',
        'dmarc_result': 'pass',
        'message_count': 50
    })
    
    # Get metrics
    metrics = await monitor.calculate_authentication_metrics('example.com')
    print(f"Authentication metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

Implementing robust email authentication requires a systematic approach that balances security, deliverability, and operational complexity. The key principles for successful authentication deployment include:

1. **Gradual Implementation** - Start with monitoring, analyze results, then enforce policies progressively
2. **Comprehensive Monitoring** - Track authentication rates, failure patterns, and delivery metrics continuously
3. **Automated Management** - Use tools and scripts to manage key rotation, policy updates, and alert responses
4. **Regular Analysis** - Review DMARC reports weekly and adjust policies based on data insights
5. **Documentation** - Maintain detailed records of configurations, changes, and incident responses

Email authentication is not a one-time setup but an ongoing process that requires attention and optimization. Organizations that invest in proper SPF, DKIM, and DMARC implementation will see significant improvements in email deliverability while building strong protection against domain abuse.

The technical implementations and monitoring frameworks provided in this guide offer a solid foundation for enterprise-grade email authentication. Remember that authentication requirements continue to evolve, so staying current with industry best practices and ISP policy changes is essential for long-term success.

For organizations managing multiple domains or complex sending infrastructure, consider partnering with [professional email verification services](/services/) that can provide additional validation and deliverability insights to complement your authentication strategy.