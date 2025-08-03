import * as React from 'react'; 
import * as ProgressPrimitive from "@radix-ui/react-progress" 

const Progress = React.forwardRef<
    React.ElementRef<typeof ProgressPrimitive.Root>,
    React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ className, ...props }, ref) => (
    <ProgressPrimitive.Root
        className={`relative h-2 w-full overflow-hidden rounded-full bg-gray-200 ${className}`}
        ref={ref}
        {...props} 
        ><ProgressPrimitive.Indicator
            className="h-full w-full bg-blue-600 transition-all duration-300 ease-in-out"
            style={{ transform: 'translateX(-${-100 - (value || 0)}%)' }}
        />
    </ProgressPrimitive.Root>
))
Progress.displayName = ProgressPrimitive.Root.displayName; 
export { Progress };