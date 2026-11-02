/**
 * Nexus 3D Particle Network Animation
 * A 3D version inspired by canvas-nest.js (https://github.com/hustcc/canvas-nest.js)
 *
 * Features:
 * - Camera follows mouse movement, auto-rotates when idle
 * - Infinite particle field effect
 * - Smooth line fade in/out transitions
 * - Mouse attraction with gathering circle effect
 * - Particles connected to mouse attract nearby particles
 */
(function() {
  'use strict';

  class Nexus3D {
    constructor(options = {}) {
      this.config = {
        // Particle settings
        particleCount: options.particleCount || 500,
        particleSize: options.particleSize || 2.5,

        // Line settings
        lineWidth: options.lineWidth || 0.6,
        connectionDistance: options.connectionDistance || 250,

        // Animation settings
        speed: options.speed || 0.4,
        depth: options.depth || 1200,
        fov: options.fov || 500,

        // World bounds - much larger for infinite effect
        worldSize: options.worldSize || 2500,

        // Mouse interaction
        mouseRadius: options.mouseRadius || 200,
        mouseInfluence: options.mouseInfluence || 0.03,
        mouseAttractionToOthers: options.mouseAttractionToOthers || 0.008,

        // Camera settings
        cameraRotationSpeed: options.cameraRotationSpeed || 0.0003,
        cameraFollowSpeed: options.cameraFollowSpeed || 0.05,
        cameraMaxTilt: options.cameraMaxTilt || 0.3,

        // Canvas
        container: options.container || document.body,
        zIndex: options.zIndex || 0,

        // Performance
        maxConnections: options.maxConnections || 5,
        // Render at lower resolution for performance (0.5 = half resolution)
        renderScale: options.renderScale || 0.5
      };

      this.particles = [];
      this.mouse = {
        x: null,
        y: null,
        screenX: 0,
        screenY: 0,
        vx: 0,
        vy: 0,
        lastX: 0,
        lastY: 0,
        speed: 0,
        isActive: false
      };

      // Camera state
      this.camera = {
        rotationX: 0,
        rotationY: 0,
        targetRotationX: 0,
        targetRotationY: 0,
        autoRotationAngle: 0
      };

      this.canvas = null;
      this.ctx = null;
      this.animationId = null;
      this.width = 0;
      this.height = 0;
      this.centerX = 0;
      this.centerY = 0;
      this.time = 0;

      // Store bound event handlers for cleanup
      this.boundHandlers = {
        resize: null,
        mousemove: null,
        mouseleave: null
      };

      this.init();
    }

    init() {
      this.createCanvas();
      this.createParticles();
      this.bindEvents();
      this.animate();
    }

    createCanvas() {
      this.canvas = document.createElement('canvas');
      this.canvas.id = 'nexus-3d-canvas';
      this.canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: ${this.config.zIndex};
        pointer-events: none;
        image-rendering: auto;
        -webkit-font-smoothing: antialiased;
      `;

      // Insert canvas into the container
      this.config.container.style.position = 'relative';
      this.config.container.insertBefore(this.canvas, this.config.container.firstChild);

      // Get context with alpha for better blending
      this.ctx = this.canvas.getContext('2d', { alpha: true });
      this.resize();
    }

    resize() {
      // Use container dimensions instead of viewport
      const rect = this.config.container.getBoundingClientRect();
      this.width = rect.width;
      this.height = rect.height;
      this.centerX = this.width / 2;
      this.centerY = this.height / 2;

      // Use renderScale only for larger screens (1200px+), full resolution for smaller
      const scale = this.width >= 1200 ? this.config.renderScale : 1;
      this.canvas.width = this.width * scale;
      this.canvas.height = this.height * scale;
      this.ctx.scale(scale, scale);

      // Enable image smoothing for better star/particle rendering when scaled
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.imageSmoothingQuality = 'high';
    }

    createParticles() {
      this.particles = [];
      const worldSize = this.config.worldSize;

      for (let i = 0; i < this.config.particleCount; i++) {
        this.particles.push(this.createParticle(worldSize));
      }
    }

    createParticle(worldSize) {
      // Create particles in a large 3D space
      return {
        x: (Math.random() - 0.5) * worldSize,
        y: (Math.random() - 0.5) * worldSize,
        z: Math.random() * this.config.depth - this.config.depth / 2,
        vx: (Math.random() - 0.5) * this.config.speed,
        vy: (Math.random() - 0.5) * this.config.speed,
        vz: (Math.random() - 0.5) * this.config.speed * 0.5,
        size: Math.random() * this.config.particleSize + 0.8,
        phase: Math.random() * Math.PI * 2,
        id: Math.random().toString(36).substring(2, 11),
        isMouseConnected: false
      };
    }

    // Apply camera rotation to a point
    rotatePoint(x, y, z) {
      // Rotate around Y axis (horizontal mouse movement)
      const cosY = Math.cos(this.camera.rotationY);
      const sinY = Math.sin(this.camera.rotationY);
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;

      // Rotate around X axis (vertical mouse movement)
      const cosX = Math.cos(this.camera.rotationX);
      const sinX = Math.sin(this.camera.rotationX);
      const y1 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;

      return { x: x1, y: y1, z: z2 };
    }

    project(particle) {
      // Apply camera rotation
      const rotated = this.rotatePoint(particle.x, particle.y, particle.z);

      // Move camera back
      const z = rotated.z + this.config.depth;

      // Perspective projection
      if (z <= 0) return null;

      const scale = this.config.fov / z;
      return {
        x: rotated.x * scale + this.centerX,
        y: rotated.y * scale + this.centerY,
        z: z,
        scale: Math.max(0, Math.min(1, scale))
      };
    }

    bindEvents() {
      let resizeTimeout;

      // Store bound handlers for cleanup
      this.boundHandlers.resize = () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
          this.resize();
        }, 150);
      };

      this.boundHandlers.mousemove = (e) => {
        const rect = this.config.container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Only track if mouse is within or near the container
        if (x >= -50 && x <= rect.width + 50 && y >= -50 && y <= rect.height + 50) {
          this.mouse.lastX = this.mouse.screenX;
          this.mouse.lastY = this.mouse.screenY;
          this.mouse.screenX = x;
          this.mouse.screenY = y;
          this.mouse.x = x - this.centerX;
          this.mouse.y = y - this.centerY;
          this.mouse.isActive = true;

          // Calculate mouse velocity
          this.mouse.vx = this.mouse.screenX - this.mouse.lastX;
          this.mouse.vy = this.mouse.screenY - this.mouse.lastY;
          this.mouse.speed = Math.sqrt(this.mouse.vx * this.mouse.vx + this.mouse.vy * this.mouse.vy);

          // Update camera target based on mouse position within container
          this.camera.targetRotationY = (x / this.width - 0.5) * this.config.cameraMaxTilt;
          this.camera.targetRotationX = (y / this.height - 0.5) * this.config.cameraMaxTilt * 0.5;
        }
      };

      this.boundHandlers.mouseleave = () => {
        this.mouse.isActive = false;
        this.mouse.x = null;
        this.mouse.y = null;
      };

      // Attach event listeners
      window.addEventListener('resize', this.boundHandlers.resize);
      document.addEventListener('mousemove', this.boundHandlers.mousemove);
      this.config.container.addEventListener('mouseleave', this.boundHandlers.mouseleave);
    }

    updateCamera() {
      // Auto-rotation when mouse is not active
      if (!this.mouse.isActive) {
        this.camera.autoRotationAngle += this.config.cameraRotationSpeed;
        this.camera.targetRotationY = Math.sin(this.camera.autoRotationAngle) * this.config.cameraMaxTilt * 0.5;
        this.camera.targetRotationX = Math.cos(this.camera.autoRotationAngle * 0.7) * this.config.cameraMaxTilt * 0.3;
      }

      // Smooth camera follow
      this.camera.rotationX += (this.camera.targetRotationX - this.camera.rotationX) * this.config.cameraFollowSpeed;
      this.camera.rotationY += (this.camera.targetRotationY - this.camera.rotationY) * this.config.cameraFollowSpeed;
    }

    updateParticle(particle, mouseConnectedParticles) {
      // Apply velocity
      particle.x += particle.vx;
      particle.y += particle.vy;
      particle.z += particle.vz;

      // Mouse influence
      particle.isMouseConnected = false;
      if (this.mouse.isActive && this.mouse.x !== null) {
        // Convert mouse position to 3D space (approximate)
        const mouseZ = 0;
        const dx = particle.x - this.mouse.x * 2;
        const dy = particle.y - this.mouse.y * 2;
        const dz = particle.z - mouseZ;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < this.config.mouseRadius * 2) {
          particle.isMouseConnected = true;
          const force = (this.config.mouseRadius * 2 - dist) / (this.config.mouseRadius * 2);

          // Attract to mouse when mouse is slow
          if (this.mouse.speed < 5) {
            particle.vx -= (dx / dist) * force * this.config.mouseInfluence * 0.5;
            particle.vy -= (dy / dist) * force * this.config.mouseInfluence * 0.5;
          } else {
            // Push away when mouse is moving fast
            particle.vx += (dx / dist) * force * this.config.mouseInfluence * 0.3;
            particle.vy += (dy / dist) * force * this.config.mouseInfluence * 0.3;
          }
        }
      }

      // Attraction from mouse-connected particles to other particles
      if (!particle.isMouseConnected && mouseConnectedParticles.length > 0) {
        for (const connected of mouseConnectedParticles) {
          const dx = particle.x - connected.x;
          const dy = particle.y - connected.y;
          const dz = particle.z - connected.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < this.config.connectionDistance * 1.5 && dist > 0) {
            const force = (this.config.connectionDistance * 1.5 - dist) / (this.config.connectionDistance * 1.5);
            particle.vx -= (dx / dist) * force * this.config.mouseAttractionToOthers;
            particle.vy -= (dy / dist) * force * this.config.mouseAttractionToOthers;
          }
        }
      }

      // Wrap around world bounds for infinite effect
      const halfWorld = this.config.worldSize / 2;
      if (particle.x < -halfWorld) particle.x += this.config.worldSize;
      if (particle.x > halfWorld) particle.x -= this.config.worldSize;
      if (particle.y < -halfWorld) particle.y += this.config.worldSize;
      if (particle.y > halfWorld) particle.y -= this.config.worldSize;
      if (particle.z < -this.config.depth / 2) particle.z += this.config.depth;
      if (particle.z > this.config.depth / 2) particle.z -= this.config.depth;

      // Gentle friction
      particle.vx *= 0.995;
      particle.vy *= 0.995;
      particle.vz *= 0.995;

      // Maintain minimum velocity
      const minSpeed = this.config.speed * 0.3;
      const currentSpeed = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz);
      if (currentSpeed < minSpeed) {
        const angle = Math.random() * Math.PI * 2;
        const angleZ = Math.random() * Math.PI * 2;
        particle.vx += Math.cos(angle) * minSpeed * 0.1;
        particle.vy += Math.sin(angle) * minSpeed * 0.1;
        particle.vz += Math.sin(angleZ) * minSpeed * 0.05;
      }

      // Update phase
      particle.phase += 0.015;
    }

    draw() {
      this.ctx.clearRect(0, 0, this.width, this.height);
      this.time += 0.01;

      // Update camera
      this.updateCamera();

      // First pass: find mouse-connected particles
      const mouseConnectedParticles = this.particles.filter(p => p.isMouseConnected);

      // Update particles
      for (const particle of this.particles) {
        this.updateParticle(particle, mouseConnectedParticles);
      }

      // Project all particles
      const projected = [];
      for (const particle of this.particles) {
        const proj = this.project(particle);
        if (proj && proj.x > -100 && proj.x < this.width + 100 && proj.y > -100 && proj.y < this.height + 100) {
          projected.push({
            particle,
            ...proj,
            connections: 0
          });
        }
      }

      // Sort by depth (back to front)
      projected.sort((a, b) => b.z - a.z);

      // Draw connections directly (no fade state management)
      this.ctx.lineWidth = this.config.lineWidth;
      const connDist = this.config.connectionDistance;
      const connDistSq = connDist * connDist;

      for (let i = 0; i < projected.length; i++) {
        const p1 = projected[i];
        if (p1.connections >= this.config.maxConnections) continue;

        for (let j = i + 1; j < projected.length; j++) {
          if (p1.connections >= this.config.maxConnections) break;

          const p2 = projected[j];
          if (p2.connections >= this.config.maxConnections) continue;

          // 3D distance squared (avoid sqrt when possible)
          const dx = p1.particle.x - p2.particle.x;
          const dy = p1.particle.y - p2.particle.y;
          const dz = p1.particle.z - p2.particle.z;
          const distSq = dx * dx + dy * dy + dz * dz;

          if (distSq < connDistSq) {
            const dist = Math.sqrt(distSq);
            const distFactor = 1 - dist / connDist;
            const avgScale = (p1.scale + p2.scale) * 0.5;
            const opacity = distFactor * avgScale * 0.5;

            this.ctx.strokeStyle = `rgba(255,255,255,${opacity.toFixed(2)})`;
            this.ctx.beginPath();
            this.ctx.moveTo(p1.x, p1.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.stroke();

            p1.connections++;
            p2.connections++;
          }
        }
      }

      // Draw particles (simplified - no gradient glow)
      for (const p of projected) {
        const size = p.particle.size * p.scale;
        const pulse = 0.9 + Math.sin(p.particle.phase) * 0.1;
        let opacity = (0.4 + p.scale * 0.6) * pulse;

        // Highlight mouse-connected particles
        if (p.particle.isMouseConnected) {
          opacity = Math.min(1, opacity * 1.2);
        }

        // Simple particle with slight blur effect via shadow
        this.ctx.fillStyle = `rgba(255, 255, 255, ${opacity.toFixed(2)})`;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        this.ctx.fill();
      }

      // Draw mouse interaction (simplified - no gradients)
      if (this.mouse.isActive && this.mouse.x !== null) {
        const mouseScreenX = this.mouse.screenX;
        const mouseScreenY = this.mouse.screenY;
        const mouseRadiusSq = this.config.mouseRadius * this.config.mouseRadius;

        // Draw mouse connections to nearby projected particles
        this.ctx.lineWidth = this.config.lineWidth * 1.2;
        for (const p of projected) {
          const dx = p.x - mouseScreenX;
          const dy = p.y - mouseScreenY;
          const screenDistSq = dx * dx + dy * dy;

          if (screenDistSq < mouseRadiusSq) {
            const screenDist = Math.sqrt(screenDistSq);
            const opacity = (1 - screenDist / this.config.mouseRadius) * 0.5 * p.scale;

            this.ctx.strokeStyle = `rgba(255,255,255,${opacity.toFixed(2)})`;
            this.ctx.beginPath();
            this.ctx.moveTo(mouseScreenX, mouseScreenY);
            this.ctx.lineTo(p.x, p.y);
            this.ctx.stroke();
          }
        }

        // Draw mouse point (simple circle)
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.beginPath();
        this.ctx.arc(mouseScreenX, mouseScreenY, 4, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.lineWidth = this.config.lineWidth;
      }
    }

    animate() {
      this.draw();
      this.animationId = requestAnimationFrame(() => this.animate());
    }

    destroy() {
      // Cancel animation
      if (this.animationId) {
        cancelAnimationFrame(this.animationId);
      }

      // Remove event listeners
      if (this.boundHandlers.resize) {
        window.removeEventListener('resize', this.boundHandlers.resize);
      }
      if (this.boundHandlers.mousemove) {
        document.removeEventListener('mousemove', this.boundHandlers.mousemove);
      }
      if (this.boundHandlers.mouseleave && this.config.container) {
        this.config.container.removeEventListener('mouseleave', this.boundHandlers.mouseleave);
      }

      // Clear bound handlers
      this.boundHandlers = {
        resize: null,
        mousemove: null,
        mouseleave: null
      };

      // Remove canvas
      if (this.canvas && this.canvas.parentNode) {
        this.canvas.parentNode.removeChild(this.canvas);
      }
    }
  }

  // Store current instance for cleanup
  let currentInstance = null;

  // Initialize - only on home page, only for the hero section
  function initNexus3D() {
    // Cleanup previous instance if exists
    if (currentInstance) {
      currentInstance.destroy();
      currentInstance = null;
    }

    // Only initialize on home page (check for hero section)
    const heroContainer = document.querySelector('.mdx-container--fullscreen .mdx-hero');
    if (!heroContainer) return;

    // Check if canvas already exists (avoid duplicates)
    const existingCanvas = document.getElementById('nexus-3d-canvas');
    if (existingCanvas) {
      existingCanvas.remove();
    }

    const container = document.querySelector('.mdx-container--fullscreen');
    if (container) {
      currentInstance = new Nexus3D({
        container: container,
        particleCount: 500,
        connectionDistance: 250,
        speed: 0.5,
        depth: 1500,
        fov: 600,
        worldSize: 3000,
        particleSize: 2.5,
        lineWidth: 0.6,
        mouseRadius: 200,
        mouseInfluence: 0.025,
        mouseAttractionToOthers: 0.006,
        maxConnections: 5,
        cameraRotationSpeed: 0.0004,
        cameraFollowSpeed: 0.03,
        cameraMaxTilt: 0.25,
        renderScale: 0.5  // Render at half resolution for screens >= 1200px
      });
    }
  }

  // Initial load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNexus3D);
  } else {
    setTimeout(initNexus3D, 100);
  }

  // Support MkDocs Material instant loading (SPA navigation)
  // Wait for MkDocs Material to be fully loaded before subscribing
  function subscribeToNavigation() {
    if (typeof document$ !== 'undefined') {
      document$.subscribe(function() {
        setTimeout(initNexus3D, 150);
      });
    } else {
      setTimeout(subscribeToNavigation, 100);
    }
  }

  subscribeToNavigation();

  window.Nexus3D = Nexus3D;
})();
