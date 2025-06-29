# Deployment Guide

## Consumer Segmentation Analytics Deployment

This guide provides comprehensive instructions for deploying the Consumer Segmentation Analytics platform across different environments.

## Deployment Options

### 1. Netlify (Recommended for Demo/Static)

#### Quick Deploy
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/your-org/consumer-segmentation)

#### Manual Deployment

1. **Prepare Repository**
   ```bash
   git clone https://github.com/your-org/consumer-segmentation.git
   cd consumer-segmentation
   ```

2. **Configure Build Settings**
   - Build command: `python build_static.py`
   - Publish directory: `dist`
   - Node version: 18
   - Python version: 3.9

3. **Environment Variables**
   ```bash
   ENVIRONMENT=production
   CACHE_ENABLED=true
   LOG_LEVEL=INFO
   ```

4. **Deploy**
   ```bash
   # Install Netlify CLI
   npm install -g netlify-cli
   
   # Login to Netlify
   netlify login
   
   # Deploy
   netlify deploy --prod --dir=dist
   ```

### 2. Docker Deployment

#### Development Environment

```bash
# Build development image
docker build -f deployment/Dockerfile --target development -t consumer-segmentation:dev .

# Run development container
docker run -p 8501:8501 -v $(pwd):/app consumer-segmentation:dev
```

#### Production Environment

```bash
# Build production image
docker build -f deployment/Dockerfile --target production -t consumer-segmentation:prod .

# Run production container
docker run -p 8501:8501 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  consumer-segmentation:prod
```

#### Docker Compose

```bash
# Development
docker-compose up consumer-segmentation-dev

# Production with full stack
docker-compose --profile production up
```

### 3. Cloud Platforms

#### AWS Deployment

1. **ECS with Fargate**
   ```bash
   # Build and push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   
   docker build -t consumer-segmentation .
   docker tag consumer-segmentation:latest <account>.dkr.ecr.us-east-1.amazonaws.com/consumer-segmentation:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/consumer-segmentation:latest
   ```

2. **ECS Task Definition**
   ```json
   {
     "family": "consumer-segmentation",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "consumer-segmentation",
         "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/consumer-segmentation:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "ENVIRONMENT",
             "value": "production"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/consumer-segmentation",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

#### Google Cloud Platform

1. **Cloud Run Deployment**
   ```bash
   # Build and push to Container Registry
   gcloud builds submit --tag gcr.io/PROJECT_ID/consumer-segmentation
   
   # Deploy to Cloud Run
   gcloud run deploy consumer-segmentation \
     --image gcr.io/PROJECT_ID/consumer-segmentation \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8501 \
     --memory 2Gi \
     --cpu 1 \
     --set-env-vars ENVIRONMENT=production
   ```

#### Azure Container Instances

```bash
# Create resource group
az group create --name consumer-segmentation-rg --location eastus

# Deploy container
az container create \
  --resource-group consumer-segmentation-rg \
  --name consumer-segmentation \
  --image your-registry/consumer-segmentation:latest \
  --dns-name-label consumer-segmentation \
  --ports 8501 \
  --environment-variables ENVIRONMENT=production \
  --cpu 1 \
  --memory 2
```

### 4. Kubernetes Deployment

#### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: consumer-segmentation
  labels:
    app: consumer-segmentation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: consumer-segmentation
  template:
    metadata:
      labels:
        app: consumer-segmentation
    spec:
      containers:
      - name: consumer-segmentation
        image: consumer-segmentation:latest
        ports:
        - containerPort: 8501
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: consumer-segmentation-service
spec:
  selector:
    app: consumer-segmentation
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
```

#### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: consumer-segmentation-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
  - hosts:
    - consumer-segmentation.yourdomain.com
    secretName: consumer-segmentation-tls
  rules:
  - host: consumer-segmentation.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: consumer-segmentation-service
            port:
              number: 80
```

## Environment Configuration

### Development Environment

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
CACHE_ENABLED=true
SAMPLE_SIZE=1000
MAX_WORKERS=2
```

### Staging Environment

```bash
# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
CACHE_ENABLED=true
SAMPLE_SIZE=10000
MAX_WORKERS=4
```

### Production Environment

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
CACHE_ENABLED=true
MAX_WORKERS=8
MEMORY_LIMIT_MB=8192
```

## Security Configuration

### SSL/TLS Setup

#### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name consumer-segmentation.yourdomain.com;

    ssl_certificate /etc/ssl/certs/consumer-segmentation.crt;
    ssl_certificate_key /etc/ssl/private/consumer-segmentation.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
```

### Authentication Setup

#### Basic Authentication

```python
# auth.py
import streamlit as st
import hashlib

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and hashlib.sha256(st.session_state["password"].encode()).hexdigest()
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True
```

## Monitoring & Logging

### Application Monitoring

#### Health Check Endpoint

```python
# health.py
import streamlit as st
from datetime import datetime
import psutil
import os

def health_check():
    """Application health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "disk_usage": psutil.disk_usage('/').percent
    }
```

#### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total app requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_duration_seconds', 'Request latency')
ACTIVE_USERS = Gauge('app_active_users', 'Number of active users')
MEMORY_USAGE = Gauge('app_memory_usage_bytes', 'Memory usage in bytes')

def track_request(func):
    """Decorator to track request metrics"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
            return result
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)
    return wrapper
```

### Logging Configuration

#### Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

def setup_logging():
    """Setup structured logging"""
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
```

## Performance Optimization

### Caching Strategy

#### Redis Configuration

```python
# cache_config.py
import redis
import pickle
import os

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=False
        )
    
    def get(self, key):
        """Get cached value"""
        data = self.redis_client.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    def set(self, key, value, ttl=3600):
        """Set cached value with TTL"""
        data = pickle.dumps(value)
        self.redis_client.setex(key, ttl, data)
    
    def delete(self, key):
        """Delete cached value"""
        self.redis_client.delete(key)
```

### Database Optimization

#### Connection Pooling

```python
# db_config.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import os

def create_db_engine():
    """Create database engine with connection pooling"""
    database_url = os.getenv('DATABASE_URL')
    
    engine = create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    return engine
```

## Backup & Recovery

### Data Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# File system backup
tar -czf data_backup_$(date +%Y%m%d_%H%M%S).tar.gz data/

# Upload to cloud storage
aws s3 cp backup_*.sql s3://your-backup-bucket/database/
aws s3 cp data_backup_*.tar.gz s3://your-backup-bucket/files/

# Cleanup old backups (keep last 30 days)
find . -name "backup_*.sql" -mtime +30 -delete
find . -name "data_backup_*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

# Download latest backup
aws s3 cp s3://your-backup-bucket/database/latest.sql ./
aws s3 cp s3://your-backup-bucket/files/latest.tar.gz ./

# Restore database
psql $DATABASE_URL < latest.sql

# Restore files
tar -xzf latest.tar.gz

echo "Restore completed successfully"
```

## Scaling Considerations

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
upstream consumer_segmentation {
    least_conn;
    server app1:8501 max_fails=3 fail_timeout=30s;
    server app2:8501 max_fails=3 fail_timeout=30s;
    server app3:8501 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://consumer_segmentation;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Session affinity
        ip_hash;
    }
}
```

### Auto-scaling

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: consumer-segmentation-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: consumer-segmentation
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues

#### Memory Issues

```bash
# Check memory usage
docker stats consumer-segmentation

# Increase memory limit
docker run --memory=4g consumer-segmentation

# Monitor memory in application
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

#### Performance Issues

```bash
# Profile application
python -m cProfile -o profile.stats app.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### Connection Issues

```bash
# Test database connection
python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Database connection successful')
conn.close()
"

# Test Redis connection
python -c "
import redis
r = redis.Redis(host='$REDIS_HOST')
r.ping()
print('Redis connection successful')
"
```

### Log Analysis

```bash
# View application logs
docker logs consumer-segmentation

# Follow logs in real-time
docker logs -f consumer-segmentation

# Search for errors
docker logs consumer-segmentation 2>&1 | grep ERROR

# Analyze log patterns
grep "ERROR" /var/log/app.log | awk '{print $1}' | sort | uniq -c
```

This deployment guide provides comprehensive instructions for deploying the Consumer Segmentation Analytics platform across various environments and platforms, with proper security, monitoring, and scaling considerations.