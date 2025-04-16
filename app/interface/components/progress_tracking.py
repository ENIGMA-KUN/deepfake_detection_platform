"""
Progress tracking components for the Deepfake Detection Platform.

This module provides components for visualizing processing progress 
with a Tron Legacy-inspired design, including progress bars, status 
indicators, and animated loaders.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import logging

# Import utilities from parent modules
from app.utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger('progress_tracking')

class TronTheme:
    """Color constants for Tron theme."""
    DEEP_BLACK = "#000000"
    DARK_TEAL = "#57A3B7"
    NEON_CYAN = "#00FFFF"
    SKY_BLUE = "#BFD4FF"
    POWDER_BLUE = "#BBFEEF"
    MAX_BLUE = "#48B3B6"
    ELECTRIC_PURPLE = "#3B1AAA"
    SUNRAY = "#DE945B"
    MAGENTA = "#B20D58"

class TronProgressBar:
    """
    Custom progress bar with Tron-themed styling.
    
    This component provides a glowing, animated progress bar with
    Tron Legacy-inspired design elements.
    """
    
    def __init__(self, parent, value=0, mode='determinate', height=20):
        """
        Initialize the progress bar.
        
        Args:
            parent: Parent widget
            value: Initial progress value (0-100)
            mode: Progress mode ('determinate' or 'indeterminate')
            height: Bar height in pixels
        """
        self.parent = parent
        self.value = max(0, min(100, value))
        self.mode = mode
        self.height = height
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create canvas for drawing progress bar
        self.canvas = tk.Canvas(
            self.frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.DARK_TEAL,
            highlightthickness=1,
            height=self.height
        )
        self.canvas.pack(fill=tk.X)
        
        # Progress label
        self.label_var = tk.StringVar(value=f"{int(self.value)}%")
        self.label = ttk.Label(
            self.frame,
            textvariable=self.label_var,
            anchor=tk.CENTER,
            background=TronTheme.DEEP_BLACK,
            foreground="white"
        )
        self.label.pack(fill=tk.X, pady=(2, 0))
        
        # Animation parameters
        self.animation_active = False
        self.glow_offset = 0
        
        # Draw initial progress
        self.draw_progress()
        
        # Start indeterminate animation if needed
        if self.mode == 'indeterminate':
            self.start_animation()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def draw_progress(self):
        """Draw the progress bar."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Get dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Default width if not drawn yet
        if width <= 1:
            width = 300
        
        # Draw background
        self.canvas.create_rectangle(
            0, 0,
            width, height,
            fill=TronTheme.DEEP_BLACK,
            outline=TronTheme.DARK_TEAL,
            width=1
        )
        
        if self.mode == 'determinate':
            # Calculate filled width
            filled_width = int(width * self.value / 100)
            
            # Draw filled portion
            if filled_width > 0:
                # Create gradient fill
                for x in range(filled_width):
                    # Position ratio
                    ratio = x / filled_width
                    
                    # Gradient from dark teal to neon cyan
                    r = int(87 + (0 - 87) * ratio)
                    g = int(163 + (255 - 163) * ratio)
                    b = int(183 + (255 - 183) * ratio)
                    
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    
                    self.canvas.create_line(
                        x, 0,
                        x, height,
                        fill=color,
                        width=1
                    )
                
                # Add glow
                glow_width = 5
                if filled_width > glow_width:
                    self.canvas.create_rectangle(
                        filled_width - glow_width, 0,
                        filled_width, height,
                        fill=TronTheme.NEON_CYAN,
                        stipple="gray50"  # Stipple pattern for glow
                    )
            
            # Update label
            self.label_var.set(f"{int(self.value)}%")
            
        else:  # indeterminate
            # Calculate position for moving element
            pos = (self.glow_offset % 100) / 100
            start_x = int(width * pos)
            end_x = min(start_x + int(width / 5), width)
            
            # Draw indeterminate bar
            if end_x > start_x:
                # Create gradient fill
                for x in range(start_x, end_x):
                    # Position ratio within bar
                    ratio = (x - start_x) / (end_x - start_x)
                    
                    # Gradient that peaks in the middle
                    if ratio < 0.5:
                        intensity = ratio * 2
                    else:
                        intensity = (1 - ratio) * 2
                    
                    # Gradient color
                    r = int(87 * (1 - intensity))
                    g = int(163 + (255 - 163) * intensity)
                    b = int(183 + (255 - 183) * intensity)
                    
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    
                    self.canvas.create_line(
                        x, 0,
                        x, height,
                        fill=color,
                        width=1
                    )
            
            # Update label for indeterminate mode
            self.label_var.set("Processing...")
    
    def set_value(self, value):
        """
        Set the progress value.
        
        Args:
            value: Progress value (0-100)
        """
        self.value = max(0, min(100, value))
        
        if self.mode == 'determinate':
            self.draw_progress()
    
    def get_value(self):
        """
        Get the current progress value.
        
        Returns:
            Current progress value
        """
        return self.value
    
    def set_mode(self, mode):
        """
        Set the progress bar mode.
        
        Args:
            mode: Mode ('determinate' or 'indeterminate')
        """
        if self.mode != mode:
            self.mode = mode
            
            if self.mode == 'indeterminate':
                self.start_animation()
            else:
                self.stop_animation()
                
            self.draw_progress()
    
    def start_animation(self):
        """Start the animation for indeterminate mode."""
        if not self.animation_active:
            self.animation_active = True
            threading.Thread(target=self._animation_loop, daemon=True).start()
    
    def stop_animation(self):
        """Stop the animation."""
        self.animation_active = False
    
    def _animation_loop(self):
        """Animation loop for indeterminate mode."""
        while self.animation_active:
            # Update glow offset
            self.glow_offset = (self.glow_offset + 2) % 100
            
            # Update UI
            try:
                self.parent.after(30, self.draw_progress)
            except Exception as e:
                logger.error(f"Error in animation loop: {e}")
                break
            
            time.sleep(0.03)

class StatusIndicator:
    """
    Status indicator for process state visualization.
    
    This component provides a visual indicator of process status with
    different colors and animations for different states.
    """
    
    def __init__(self, parent, status='pending', size=10):
        """
        Initialize the status indicator.
        
        Args:
            parent: Parent widget
            status: Initial status ('pending', 'processing', 'complete', 'failed')
            size: Indicator size in pixels
        """
        self.parent = parent
        self.status = status
        self.size = size
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create canvas for drawing indicator
        self.canvas = tk.Canvas(
            self.frame,
            bg=TronTheme.DEEP_BLACK,
            highlightthickness=0,
            width=self.size + 6,
            height=self.size + 6
        )
        self.canvas.pack(side=tk.LEFT)
        
        # Status text
        self.label_var = tk.StringVar(value=self.get_status_text())
        self.label = ttk.Label(
            self.frame,
            textvariable=self.label_var,
            background=TronTheme.DEEP_BLACK,
            foreground=self.get_status_color()
        )
        self.label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Animation parameters
        self.animation_active = False
        self.pulse_size = 0
        
        # Draw initial indicator
        self.draw_indicator()
        
        # Start animation if processing
        if self.status == 'processing':
            self.start_animation()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def draw_indicator(self):
        """Draw the status indicator."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Get status color
        color = self.get_status_color()
        
        # Draw indicator based on status
        center_x = (self.size + 6) // 2
        center_y = (self.size + 6) // 2
        
        if self.status == 'processing':
            # Draw pulsing circle with outer glow
            # Base circle
            self.canvas.create_oval(
                center_x - self.size // 2, center_y - self.size // 2,
                center_x + self.size // 2, center_y + self.size // 2,
                fill=color,
                outline=""
            )
            
            # Pulsing glow
            if self.pulse_size > 0:
                glow_size = self.size + self.pulse_size
                self.canvas.create_oval(
                    center_x - glow_size // 2, center_y - glow_size // 2,
                    center_x + glow_size // 2, center_y + glow_size // 2,
                    fill="",
                    outline=color,
                    width=1,
                    stipple="gray25"  # Stipple pattern for glow
                )
        
        elif self.status == 'complete':
            # Draw checkmark
            self.canvas.create_oval(
                center_x - self.size // 2, center_y - self.size // 2,
                center_x + self.size // 2, center_y + self.size // 2,
                fill=color,
                outline="white",
                width=1
            )
            
            # Draw checkmark inside
            self.canvas.create_line(
                center_x - self.size // 4, center_y,
                center_x - self.size // 8, center_y + self.size // 4,
                center_x + self.size // 3, center_y - self.size // 4,
                fill="white",
                width=1
            )
        
        elif self.status == 'failed':
            # Draw X mark
            self.canvas.create_oval(
                center_x - self.size // 2, center_y - self.size // 2,
                center_x + self.size // 2, center_y + self.size // 2,
                fill=color,
                outline="white",
                width=1
            )
            
            # Draw X inside
            self.canvas.create_line(
                center_x - self.size // 4, center_y - self.size // 4,
                center_x + self.size // 4, center_y + self.size // 4,
                fill="white",
                width=1
            )
            self.canvas.create_line(
                center_x - self.size // 4, center_y + self.size // 4,
                center_x + self.size // 4, center_y - self.size // 4,
                fill="white",
                width=1
            )
        
        else:  # pending
            # Draw hollow circle
            self.canvas.create_oval(
                center_x - self.size // 2, center_y - self.size // 2,
                center_x + self.size // 2, center_y + self.size // 2,
                fill="",
                outline=color,
                width=1
            )
    
    def set_status(self, status):
        """
        Set the indicator status.
        
        Args:
            status: Status ('pending', 'processing', 'complete', 'failed')
        """
        if self.status != status:
            self.status = status
            self.label_var.set(self.get_status_text())
            self.label.config(foreground=self.get_status_color())
            
            # Start or stop animation based on status
            if self.status == 'processing':
                self.start_animation()
            else:
                self.stop_animation()
                
            self.draw_indicator()
    
    def get_status_color(self):
        """
        Get color for current status.
        
        Returns:
            Color hex code
        """
        if self.status == 'pending':
            return TronTheme.DARK_TEAL
        elif self.status == 'processing':
            return TronTheme.SUNRAY
        elif self.status == 'complete':
            return TronTheme.NEON_CYAN
        elif self.status == 'failed':
            return TronTheme.MAGENTA
        else:
            return "white"
    
    def get_status_text(self):
        """
        Get text for current status.
        
        Returns:
            Status text
        """
        if self.status == 'pending':
            return "Pending"
        elif self.status == 'processing':
            return "Processing"
        elif self.status == 'complete':
            return "Complete"
        elif self.status == 'failed':
            return "Failed"
        else:
            return "Unknown"
    
    def start_animation(self):
        """Start the animation for processing status."""
        if not self.animation_active:
            self.animation_active = True
            threading.Thread(target=self._animation_loop, daemon=True).start()
    
    def stop_animation(self):
        """Stop the animation."""
        self.animation_active = False
        self.pulse_size = 0
    
    def _animation_loop(self):
        """Animation loop for processing status."""
        pulse_direction = 1
        
        while self.animation_active:
            # Update pulse size
            self.pulse_size += pulse_direction
            
            # Reverse direction at limits
            if self.pulse_size >= 6:
                pulse_direction = -1
            elif self.pulse_size <= 0:
                pulse_direction = 1
            
            # Update UI
            try:
                self.parent.after(50, self.draw_indicator)
            except Exception as e:
                logger.error(f"Error in animation loop: {e}")
                break
            
            time.sleep(0.05)

class ProcessingQueue:
    """
    Component for displaying queued processing tasks.
    
    This component visualizes the current processing queue with status
    indicators and progress tracking.
    """
    
    def __init__(self, parent):
        """
        Initialize the processing queue component.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        
        # Create frame with Tron styling
        self.frame = ttk.Frame(parent)
        
        # Create header
        header_frame = ttk.Frame(self.frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame,
            text="Processing Queue",
            style="Subheader.TLabel"
        )
        title_label.pack(side=tk.LEFT)
        
        # Queue stats
        self.stats_var = tk.StringVar(value="0 items")
        stats_label = ttk.Label(
            header_frame,
            textvariable=self.stats_var
        )
        stats_label.pack(side=tk.RIGHT)
        
        # Create list frame with border
        list_frame = ttk.Frame(
            self.frame,
            borderwidth=1,
            relief=tk.SOLID
        )
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar for item list
        self.canvas = tk.Canvas(
            list_frame,
            bg=TronTheme.DEEP_BLACK,
            highlightbackground=TronTheme.NEON_CYAN,
            highlightthickness=1
        )
        
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient=tk.VERTICAL,
            command=self.canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas for items
        self.item_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            0, 0,
            window=self.item_frame,
            anchor=tk.NW,
            width=self.canvas.winfo_width()
        )
        
        # Bind events for scrolling and resizing
        self.item_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Queue items
        self.queue_items = []
        
        # Update for empty state
        self.update_empty_state()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def _on_frame_configure(self, event):
        """Handle frame configuration events."""
        # Update the scrollregion to include the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize events."""
        # Update the width of the inner frame to fill the canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def add_item(self, item_id, filename, media_type, status='pending', progress=0):
        """
        Add an item to the processing queue.
        
        Args:
            item_id: Unique identifier for the item
            filename: Name of the file
            media_type: Type of media ('image', 'audio', 'video')
            status: Initial status ('pending', 'processing', 'complete', 'failed')
            progress: Initial progress (0-100)
            
        Returns:
            Item frame
        """
        # Check if item already exists
        for item in self.queue_items:
            if item['id'] == item_id:
                # Update existing item
                self.update_item(item_id, status, progress)
                return item['frame']
        
        # Create item frame
        item_frame = ttk.Frame(self.item_frame)
        item_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create item contents
        # Status indicator
        status_indicator = StatusIndicator(item_frame, status)
        status_indicator.pack(side=tk.LEFT, padx=(0, 10))
        
        # File info
        info_frame = ttk.Frame(item_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        filename_label = ttk.Label(
            info_frame,
            text=filename,
            font=("Helvetica", 10, "bold")
        )
        filename_label.pack(anchor=tk.W)
        
        type_label = ttk.Label(
            info_frame,
            text=f"Type: {media_type.capitalize()}"
        )
        type_label.pack(anchor=tk.W)
        
        # Progress bar
        progress_frame = ttk.Frame(item_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        progress_bar = TronProgressBar(progress_frame, value=progress, height=15)
        progress_bar.pack(fill=tk.X)
        
        # Store item data
        item_data = {
            'id': item_id,
            'frame': item_frame,
            'filename': filename,
            'media_type': media_type,
            'status_indicator': status_indicator,
            'progress_bar': progress_bar,
            'status': status,
            'progress': progress
        }
        
        self.queue_items.append(item_data)
        
        # Update queue stats
        self.update_stats()
        
        # Clear empty state if needed
        if len(self.queue_items) == 1:
            self.clear_empty_state()
        
        return item_frame
    
    def update_item(self, item_id, status=None, progress=None):
        """
        Update an item in the processing queue.
        
        Args:
            item_id: Unique identifier for the item
            status: New status (optional)
            progress: New progress (optional)
            
        Returns:
            True if item was updated, False if not found
        """
        # Find item
        for item in self.queue_items:
            if item['id'] == item_id:
                # Update status if provided
                if status is not None and status != item['status']:
                    item['status'] = status
                    item['status_indicator'].set_status(status)
                
                # Update progress if provided
                if progress is not None and progress != item['progress']:
                    item['progress'] = progress
                    
                    # Update progress bar
                    if status == 'processing':
                        # Use indeterminate mode if in processing with 0 progress
                        if progress == 0:
                            item['progress_bar'].set_mode('indeterminate')
                        else:
                            item['progress_bar'].set_mode('determinate')
                            item['progress_bar'].set_value(progress)
                    else:
                        item['progress_bar'].set_mode('determinate')
                        item['progress_bar'].set_value(progress)
                
                return True
        
        return False
    
    def remove_item(self, item_id):
        """
        Remove an item from the processing queue.
        
        Args:
            item_id: Unique identifier for the item
            
        Returns:
            True if item was removed, False if not found
        """
        # Find item
        for i, item in enumerate(self.queue_items):
            if item['id'] == item_id:
                # Destroy frame
                item['frame'].destroy()
                
                # Remove from list
                self.queue_items.pop(i)
                
                # Update queue stats
                self.update_stats()
                
                # Show empty state if needed
                if len(self.queue_items) == 0:
                    self.update_empty_state()
                
                return True
        
        return False
    
    def clear_queue(self):
        """Clear all items from the queue."""
        # Remove all items
        for item in self.queue_items:
            item['frame'].destroy()
        
        self.queue_items = []
        
        # Update queue stats
        self.update_stats()
        
        # Show empty state
        self.update_empty_state()
    
    def update_stats(self):
        """Update queue statistics."""
        total = len(self.queue_items)
        pending = sum(1 for item in self.queue_items if item['status'] == 'pending')
        processing = sum(1 for item in self.queue_items if item['status'] == 'processing')
        complete = sum(1 for item in self.queue_items if item['status'] == 'complete')
        failed = sum(1 for item in self.queue_items if item['status'] == 'failed')
        
        stats_text = f"{total} item{'s' if total != 1 else ''}"
        
        if total > 0:
            stats_text += f" ({pending} pending, {processing} processing, {complete} complete, {failed} failed)"
        
        self.stats_var.set(stats_text)
    
    def update_empty_state(self):
        """Show empty state message."""
        # Clear existing items
        for widget in self.item_frame.winfo_children():
            widget.destroy()
        
        # Create empty state message
        empty_frame = ttk.Frame(self.item_frame)
        empty_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        empty_label = ttk.Label(
            empty_frame,
            text="No items in processing queue",
            foreground=TronTheme.SKY_BLUE,
            font=("Helvetica", 12),
            background=TronTheme.DEEP_BLACK
        )
        empty_label.pack(pady=20)
        
        # Keep reference to prevent garbage collection
        self.empty_frame = empty_frame
    
    def clear_empty_state(self):
        """Clear the empty state message."""
        if hasattr(self, 'empty_frame'):
            self.empty_frame.destroy()
            delattr(self, 'empty_frame')

class AnimatedLoader:
    """
    Animated loading indicator with Tron theme.
    
    This component provides a visually appealing loading animation
    for use during processing operations.
    """
    
    def __init__(self, parent, size=50, message="Processing"):
        """
        Initialize the animated loader.
        
        Args:
            parent: Parent widget
            size: Size of the loader in pixels
            message: Loading message
        """
        self.parent = parent
        self.size = size
        self.message = message
        
        # Create frame
        self.frame = ttk.Frame(parent)
        
        # Create canvas for animation
        self.canvas = tk.Canvas(
            self.frame,
            bg=TronTheme.DEEP_BLACK,
            highlightthickness=0,
            width=self.size,
            height=self.size
        )
        self.canvas.pack(pady=(0, 10))
        
        # Message label
        self.label_var = tk.StringVar(value=self.message)
        self.label = ttk.Label(
            self.frame,
            textvariable=self.label_var,
            font=("Helvetica", 12),
            foreground=TronTheme.NEON_CYAN,
            background=TronTheme.DEEP_BLACK
        )
        self.label.pack()
        
        # Animation parameters
        self.animation_active = False
        self.rotation = 0
        self.dots = 0
        self.dot_timer = 0
        
        # Draw initial state
        self.draw_loader()
    
    def pack(self, **kwargs):
        """Pack the frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the frame."""
        self.frame.grid(**kwargs)
    
    def draw_loader(self):
        """Draw the loader animation."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate center
        center_x = self.size // 2
        center_y = self.size // 2
        
        # Draw outer circle
        self.canvas.create_oval(
            center_x - self.size // 2, center_y - self.size // 2,
            center_x + self.size // 2, center_y + self.size // 2,
            outline=TronTheme.DARK_TEAL,
            width=1
        )
        
        # Draw inner circle
        inner_size = self.size * 0.7
        self.canvas.create_oval(
            center_x - inner_size // 2, center_y - inner_size // 2,
            center_x + inner_size // 2, center_y + inner_size // 2,
            outline=TronTheme.DARK_TEAL,
            width=1
        )
        
        # Draw rotating elements
        for i in range(12):
            # Calculate angle with rotation offset
            angle = math.radians(i * 30 + self.rotation)
            
            # Calculate position on circle
            outer_radius = self.size // 2
            inner_radius = inner_size // 2
            
            outer_x = center_x + int(outer_radius * math.cos(angle))
            outer_y = center_y + int(outer_radius * math.sin(angle))
            
            inner_x = center_x + int(inner_radius * math.cos(angle))
            inner_y = center_y + int(inner_radius * math.sin(angle))
            
            # Gradient from cyan to dark based on position
            # Brighter near the current rotation point
            distance = abs(i * 30 - (self.rotation % 360))
            if distance > 180:
                distance = 360 - distance
            
            intensity = 1 - (distance / 180)
            
            # Color interpolation from dark teal to neon cyan
            r = int(87 + (0 - 87) * intensity)
            g = int(163 + (255 - 163) * intensity)
            b = int(183 + (255 - 183) * intensity)
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Draw line
            self.canvas.create_line(
                inner_x, inner_y,
                outer_x, outer_y,
                fill=color,
                width=2
            )
        
        # Update message with animated dots
        dots_text = "." * self.dots
        self.label_var.set(f"{self.message}{dots_text}")
    
    def start_animation(self):
        """Start the loader animation."""
        if not self.animation_active:
            self.animation_active = True
            threading.Thread(target=self._animation_loop, daemon=True).start()
    
    def stop_animation(self):
        """Stop the loader animation."""
        self.animation_active = False
    
    def set_message(self, message):
        """
        Set the loading message.
        
        Args:
            message: New message
        """
        self.message = message
    
    def _animation_loop(self):
        """Animation loop for the loader."""
        while self.animation_active:
            # Update rotation
            self.rotation = (self.rotation + 5) % 360
            
            # Update dots
            self.dot_timer += 1
            if self.dot_timer >= 10:
                self.dot_timer = 0
                self.dots = (self.dots + 1) % 4
            
            # Update UI
            try:
                self.parent.after(50, self.draw_loader)
            except Exception as e:
                logger.error(f"Error in animation loop: {e}")
                break
            
            time.sleep(0.05)