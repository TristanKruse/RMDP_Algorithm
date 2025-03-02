class DemandPattern:
    """Configurable class for time-dependent demand patterns."""
    
    def __init__(self, pattern_config=None):
        """
        Initialize with a demand pattern configuration.
        
        Args:
            pattern_config: Can be one of:
                - None: Uses default uniform demand
                - Dict with 'hourly_rates': list of 24 hourly rates
                - Dict with 'custom_periods': list of (start_time_percent, end_time_percent, rate)
                - Dict with 'function': callable that maps time_percent to rate
        """
        self.pattern_config = pattern_config or {'type': 'uniform', 'rate': 1.0}
        self.pattern_type = self.pattern_config.get('type', 'uniform')
        
        # Validate and process config based on pattern type
        if self.pattern_type == 'hourly':
            self._validate_hourly_rates()
        elif self.pattern_type == 'custom_periods':
            self._validate_custom_periods()
        elif self.pattern_type == 'function':
            self._validate_function()
    
    def _validate_hourly_rates(self):
        """Validate hourly rates configuration."""
        hourly_rates = self.pattern_config.get('hourly_rates', [])
        if not hourly_rates or len(hourly_rates) != 24:
            raise ValueError("Hourly rates must contain exactly 24 values")
    
    def _validate_custom_periods(self):
        """Validate custom periods configuration."""
        periods = self.pattern_config.get('custom_periods', [])
        if not periods:
            raise ValueError("Custom periods must not be empty")
        for period in periods:
            if len(period) != 3:
                raise ValueError("Each period must contain (start_percent, end_percent, rate)")
            start, end, rate = period
            if not (0 <= start < end <= 1) or rate < 0:
                raise ValueError("Invalid period values")
    
    def _validate_function(self):
        """Validate function configuration."""
        func = self.pattern_config.get('function')
        if not callable(func):
            raise ValueError("Function must be callable")
    
    def get_rate(self, current_time, simulation_duration):
        """
        Get the arrival rate for the current simulation time.
        
        Args:
            current_time: Current time in the simulation
            simulation_duration: Total duration of the simulation
            
        Returns:
            The current arrival rate
        """
        # Convert current time to a percentage of simulation duration (0.0 to 1.0)
        time_percent = current_time / simulation_duration
        
        if self.pattern_type == 'uniform':
            return self.pattern_config.get('rate', 1.0)
        
        elif self.pattern_type == 'hourly':
            # Map time percentage to 0-23 hour range
            hour = int(time_percent * 24) % 24
            return self.pattern_config['hourly_rates'][hour]
        
        elif self.pattern_type == 'custom_periods':
            # Find which period we're in
            for start, end, rate in self.pattern_config['custom_periods']:
                if start <= time_percent < end:
                    return rate
            return 0.0  # Default if not in any period
        
        elif self.pattern_type == 'function':
            # Call the provided function with time percentage
            return self.pattern_config['function'](time_percent)
        
        return 1.0  # Default fallback