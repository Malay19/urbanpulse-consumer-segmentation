#!/bin/bash

# Deployment script for Consumer Segmentation Analysis
set -e

echo "🚀 Deploying Consumer Segmentation Analysis"
echo "==========================================="

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
REGISTRY=${DOCKER_REGISTRY:-localhost:5000}
IMAGE_NAME="consumer-segmentation"

echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Registry: $REGISTRY"

# Validate environment
case "$ENVIRONMENT" in
    "development"|"staging"|"production")
        echo "✅ Valid environment: $ENVIRONMENT"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "Valid options: development, staging, production"
        exit 1
        ;;
esac

# Build Docker image
echo ""
echo "🏗️  Building Docker image..."
docker build -f deployment/Dockerfile -t $IMAGE_NAME:$VERSION --target $ENVIRONMENT .

# Tag image for registry
if [ "$REGISTRY" != "localhost:5000" ]; then
    echo "🏷️  Tagging image for registry..."
    docker tag $IMAGE_NAME:$VERSION $REGISTRY/$IMAGE_NAME:$VERSION
    docker tag $IMAGE_NAME:$VERSION $REGISTRY/$IMAGE_NAME:latest
fi

# Run tests in container
echo ""
echo "🧪 Running tests in container..."
docker run --rm $IMAGE_NAME:$VERSION python -m pytest test_suite.py::TestDataQuality::test_data_completeness -v

# Push to registry (if not local)
if [ "$REGISTRY" != "localhost:5000" ]; then
    echo ""
    echo "📤 Pushing to registry..."
    docker push $REGISTRY/$IMAGE_NAME:$VERSION
    docker push $REGISTRY/$IMAGE_NAME:latest
fi

# Deploy based on environment
case "$ENVIRONMENT" in
    "development")
        echo ""
        echo "🛠️  Deploying to development..."
        docker-compose -f deployment/docker-compose.yml up -d consumer-segmentation-dev
        ;;
    
    "staging")
        echo ""
        echo "🎭 Deploying to staging..."
        # Stop existing containers
        docker-compose -f deployment/docker-compose.yml down consumer-segmentation-prod || true
        
        # Deploy new version
        docker-compose -f deployment/docker-compose.yml up -d consumer-segmentation-prod
        
        # Wait for health check
        echo "⏳ Waiting for application to be healthy..."
        timeout 60 bash -c 'until curl -f http://localhost:8501/_stcore/health; do sleep 2; done'
        ;;
    
    "production")
        echo ""
        echo "🌟 Deploying to production..."
        
        # Backup current deployment
        echo "💾 Creating backup..."
        docker-compose -f deployment/docker-compose.yml exec consumer-segmentation-prod tar -czf /app/backup-$(date +%Y%m%d_%H%M%S).tar.gz /app/data /app/logs || true
        
        # Rolling deployment
        echo "🔄 Performing rolling deployment..."
        
        # Start new container
        docker-compose -f deployment/docker-compose.yml up -d --no-deps consumer-segmentation-prod
        
        # Wait for health check
        echo "⏳ Waiting for application to be healthy..."
        timeout 120 bash -c 'until curl -f http://localhost:8501/_stcore/health; do sleep 5; done'
        
        # Start nginx if not running
        docker-compose -f deployment/docker-compose.yml --profile production up -d nginx
        ;;
esac

# Verify deployment
echo ""
echo "✅ Deployment verification..."

case "$ENVIRONMENT" in
    "development")
        PORT=8501
        ;;
    "staging")
        PORT=8501
        ;;
    "production")
        PORT=80
        ;;
esac

# Check if application is responding
if curl -f http://localhost:$PORT/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application is healthy and responding on port $PORT"
else
    echo "❌ Application health check failed"
    echo "Checking logs..."
    docker-compose -f deployment/docker-compose.yml logs --tail=20 consumer-segmentation-$ENVIRONMENT || docker-compose -f deployment/docker-compose.yml logs --tail=20 consumer-segmentation-prod
    exit 1
fi

# Show deployment info
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Deployment Information:"
echo "========================="
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "URL: http://localhost:$PORT"
echo "Health Check: http://localhost:$PORT/_stcore/health"
echo ""
echo "📊 Container Status:"
docker-compose -f deployment/docker-compose.yml ps

echo ""
echo "📝 Next Steps:"
echo "- Monitor logs: docker-compose -f deployment/docker-compose.yml logs -f"
echo "- Scale if needed: docker-compose -f deployment/docker-compose.yml up -d --scale consumer-segmentation-prod=2"
echo "- Rollback if needed: docker-compose -f deployment/docker-compose.yml down && docker-compose -f deployment/docker-compose.yml up -d"