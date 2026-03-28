import React, { useState, useEffect, useMemo } from 'react';

import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from '@/components/ui/command';
import { Search, ChevronDown, Wifi, WifiOff } from 'lucide-react';

function formatVolume(v) {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

export default function MarketGrid({
  markets = [],
  value,
  onChange,
  loading = false,
  offline = false,
}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const grouped = useMemo(() => {
    const g = {};
    for (const m of markets) {
      const cat = m.category || 'Other';
      if (!g[cat]) g[cat] = [];
      g[cat].push(m);
    }
    return g;
  }, [markets]);

  const currentLabel = useMemo(() => {
    const m = markets.find((x) => x.symbol === value);
    return m ? `${m.base}/USDT` : value.replace(':USDT', '');
  }, [markets, value]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="min-w-[180px] justify-between gap-2 font-mono text-sm glass-effect"
        >
          <div className="flex items-center gap-2">
            <Search className="h-3.5 w-3.5 text-muted-foreground" />
            <span>{currentLabel}</span>
          </div>
          <div className="flex items-center gap-1.5">
            {offline ? (
              <WifiOff className="h-3 w-3 text-yellow-500" />
            ) : (
              <Wifi className="h-3 w-3 text-green-500" />
            )}
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[340px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search assets... (Cmd+K)" />
          <CommandList>
            {loading ? (
              <div className="py-6 text-center text-sm text-muted-foreground">
                Loading markets...
              </div>
            ) : (
              <>
                <CommandEmpty>No assets found.</CommandEmpty>
                {Object.entries(grouped).map(([cat, items]) => (
                  <CommandGroup key={cat} heading={cat}>
                    {items.map((m) => (
                      <CommandItem
                        key={m.symbol}
                        value={`${m.base} ${m.name} ${m.symbol}`}
                        onSelect={() => {
                          onChange(m.symbol);
                          setOpen(false);
                        }}
                        className="flex items-center justify-between"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium text-sm">
                            {m.base}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            /USDT
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          {m.price > 0 && (
                            <span>
                              $
                              {m.price >= 1
                                ? m.price.toLocaleString(undefined, {
                                    maximumFractionDigits: 2,
                                  })
                                : m.price.toPrecision(4)}
                            </span>
                          )}
                          {m.volume_24h > 0 && (
                            <span className="w-16 text-right">
                              {formatVolume(m.volume_24h)}
                            </span>
                          )}
                        </div>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                ))}
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
