//go:build darwin

package statusbar

/*
#cgo darwin CFLAGS: -x objective-c -fobjc-arc
#cgo darwin LDFLAGS: -framework Cocoa
#import <Cocoa/Cocoa.h>
#include <signal.h>
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>

static uint64_t pid_phys_footprint(pid_t pid) {
    if (pid <= 0) {
        return 0;
    }
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&ri) == 0) {
        return ri.ri_phys_footprint;
    }
    return 0;
}

static uint64_t total_decision_memory_bytes(pid_t workerPid) {
    uint64_t total = 0;
    pid_t selfPid = getpid();
    total += pid_phys_footprint(selfPid);
    if (workerPid > 0 && workerPid != selfPid) {
        total += pid_phys_footprint(workerPid);
    }

    int numPids = proc_listallpids(NULL, 0);
    if (numPids <= 0) {
        return total;
    }
    pid_t *pids = (pid_t *)calloc((size_t)numPids, sizeof(pid_t));
    if (!pids) {
        return total;
    }
    int actual = proc_listallpids(pids, numPids * (int)sizeof(pid_t));
    int count = actual / (int)sizeof(pid_t);
    char pathBuf[PROC_PIDPATHINFO_MAXSIZE];
    for (int i = 0; i < count; i++) {
        pid_t p = pids[i];
        if (p <= 0 || p == selfPid || p == workerPid) {
            continue;
        }
        if (proc_pidpath(p, pathBuf, sizeof(pathBuf)) > 0) {
            if (strstr(pathBuf, "decision-cli") != NULL) {
                total += pid_phys_footprint(p);
            }
        }
    }
    free(pids);
    return total;
}

@interface StatusBarApp : NSObject <NSApplicationDelegate>
@property (strong) NSStatusItem *statusItem;
@property (strong) NSMenu *statusMenu;
@property (strong) NSMenuItem *itemEngine;
@property (strong) NSMenuItem *itemModel;
@property (strong) NSMenuItem *itemQueueSpeed;
@property (strong) NSMenuItem *itemMemory;
@property (strong) NSTimer *timer;
@property (copy) NSString *currentSymbol;
@property (assign) int missingTicks;
@property (assign) BOOL hasSeenWorker;

- (void)setSymbolIcon:(NSString *)symbolName;
- (void)onTick:(NSTimer *)timer;
- (void)openDir:(id)sender;
- (void)openLog:(id)sender;
- (void)killWorker:(id)sender;
@end

@implementation StatusBarApp

- (void)setSymbolIcon:(NSString *)symbolName {
    if ([self.currentSymbol isEqualToString:symbolName]) {
        return;
    }
    self.currentSymbol = symbolName;
    NSImage *icon = [NSImage imageWithSystemSymbolName:symbolName accessibilityDescription:@"decision-cli"];
    if (!icon) {
        icon = [NSImage imageWithSystemSymbolName:@"arrow.left.arrow.right.square" accessibilityDescription:@"decision-cli"];
    }
    if (icon) {
        [icon setTemplate:YES];
        self.statusItem.button.image = icon;
        self.statusItem.button.title = @"";
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

    self.missingTicks = 0;
    self.hasSeenWorker = NO;

    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSSquareStatusItemLength];
    self.statusItem.button.toolTip = @"decision-cli";
    [self setSymbolIcon:@"arrow.left.arrow.right.square.fill"];

    self.statusMenu = [[NSMenu alloc] initWithTitle:@"decision-cli"];

    self.itemEngine = [[NSMenuItem alloc] initWithTitle:@"MLX 引擎: 检测中..." action:nil keyEquivalent:@""];
    [self.itemEngine setEnabled:NO];
    [self.statusMenu addItem:self.itemEngine];

    self.itemModel = [[NSMenuItem alloc] initWithTitle:@"当前模型: 无" action:nil keyEquivalent:@""];
    [self.itemModel setEnabled:NO];
    [self.statusMenu addItem:self.itemModel];

    self.itemQueueSpeed = [[NSMenuItem alloc] initWithTitle:@"排队任务: 0  |  推理速度: --" action:nil keyEquivalent:@""];
    [self.itemQueueSpeed setEnabled:NO];
    [self.statusMenu addItem:self.itemQueueSpeed];

    self.itemMemory = [[NSMenuItem alloc] initWithTitle:@"总内存占用: --" action:nil keyEquivalent:@""];
    [self.itemMemory setEnabled:NO];
    [self.statusMenu addItem:self.itemMemory];

    [self.statusMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *openDirItem = [[NSMenuItem alloc] initWithTitle:@"打开缓存目录" action:@selector(openDir:) keyEquivalent:@""];
    [openDirItem setTarget:self];
    [self.statusMenu addItem:openDirItem];

    NSMenuItem *openLogItem = [[NSMenuItem alloc] initWithTitle:@"打开日志" action:@selector(openLog:) keyEquivalent:@""];
    [openLogItem setTarget:self];
    [self.statusMenu addItem:openLogItem];

    NSMenuItem *killItem = [[NSMenuItem alloc] initWithTitle:@"退出" action:@selector(killWorker:) keyEquivalent:@"q"];
    [killItem setTarget:self];
    [self.statusMenu addItem:killItem];

    self.statusItem.menu = self.statusMenu;

    [self onTick:nil];

    self.timer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                  target:self
                                                selector:@selector(onTick:)
                                                userInfo:nil
                                                 repeats:YES];
    [[NSRunLoop currentRunLoop] addTimer:self.timer forMode:NSRunLoopCommonModes];
}

- (NSString *)formatBytes:(uint64_t)bytes {
    double mb = (double)bytes / (1024.0 * 1024.0);
    if (mb >= 1024.0) {
        return [NSString stringWithFormat:@"%.2f GB (%.0f MB)", mb / 1024.0, mb];
    }
    return [NSString stringWithFormat:@"%.0f MB", mb];
}

- (void)onTick:(NSTimer *)timer {
    NSString *statusFile = [@"~/.cache/decision-cli/worker_status.json" stringByExpandingTildeInPath];
    NSData *data = [NSData dataWithContentsOfFile:statusFile];
    NSDictionary *dict = nil;
    BOOL workerAlive = NO;
    pid_t workerPid = 0;

    if (data) {
        NSError *err = nil;
        dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (!err && dict) {
            NSNumber *pidNum = dict[@"pid"];
            if (pidNum && [pidNum intValue] > 0) {
                workerPid = (pid_t)[pidNum intValue];
                if (kill(workerPid, 0) == 0) {
                    workerAlive = YES;
                }
            }
        }
    }

    uint64_t memBytes = total_decision_memory_bytes(workerAlive ? workerPid : 0);
    [self.itemMemory setTitle:[NSString stringWithFormat:@"总内存占用: %@", [self formatBytes:memBytes]]];

    if (!workerAlive) {
        self.missingTicks++;
        int maxWait = self.hasSeenWorker ? 3 : 15;
        if (self.missingTicks >= maxWait) {
            [NSApp terminate:nil];
            return;
        }
        [self.itemEngine setTitle:@"MLX 引擎: 未运行 (未加载模型)"];
        [self.itemModel setTitle:@"当前模型: 无"];
        [self.itemQueueSpeed setTitle:@"排队任务: 0  |  推理速度: --"];
        return;
    }

    self.missingTicks = 0;
    self.hasSeenWorker = YES;

    NSString *state = dict[@"state"] ?: @"idle";
    NSString *modelPath = dict[@"current_model_path"] ?: @"";
    NSString *alias = dict[@"current_model_alias"] ?: @"";
    BOOL modelLoaded = ([modelPath length] > 0) && ![alias isEqualToString:@"空闲"] && ![alias isEqualToString:@"未加载"] && ([alias length] > 0);

    NSNumber *queue = dict[@"queue_length"] ?: @0;
    NSNumber *tps = dict[@"throughput_dps"] ?: @0.0;
    NSNumber *avgLat = dict[@"avg_latency_ms"] ?: @0.0;

    if (!modelLoaded) {
        [self.itemEngine setTitle:@"MLX 引擎: 已就绪 (未加载模型)"];
        [self.itemModel setTitle:@"当前模型: 无"];
    } else if ([state isEqualToString:@"inferring"]) {
        [self.itemEngine setTitle:@"MLX 引擎: 运行中 (已加载模型 · 推理中)"];
        [self.itemModel setTitle:[NSString stringWithFormat:@"当前模型: %@", alias]];
    } else if ([state isEqualToString:@"switching"] || [state isEqualToString:@"draining"]) {
        [self.itemEngine setTitle:@"MLX 引擎: 运行中 (正在切换模型)"];
        [self.itemModel setTitle:[NSString stringWithFormat:@"当前模型: %@", alias]];
    } else {
        [self.itemEngine setTitle:@"MLX 引擎: 运行中 (已加载模型)"];
        [self.itemModel setTitle:[NSString stringWithFormat:@"当前模型: %@", alias]];
    }

    if ([tps doubleValue] > 0.0) {
        [self.itemQueueSpeed setTitle:[NSString stringWithFormat:@"排队任务: %@  |  推理速度: %.1f 条/s (%.0f ms)", queue, [tps doubleValue], [avgLat doubleValue]]];
    } else {
        [self.itemQueueSpeed setTitle:[NSString stringWithFormat:@"排队任务: %@  |  推理速度: --", queue]];
    }
}

- (void)openDir:(id)sender {
    NSString *path = [@"~/.cache/decision-cli" stringByExpandingTildeInPath];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:path]];
}

- (void)openLog:(id)sender {
    NSString *path = [@"~/.cache/decision-cli/worker.log" stringByExpandingTildeInPath];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:path]];
}

- (void)killWorker:(id)sender {
    NSString *statusFile = [@"~/.cache/decision-cli/worker_status.json" stringByExpandingTildeInPath];
    NSData *data = [NSData dataWithContentsOfFile:statusFile];
    if (data) {
        NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        NSNumber *pidNum = dict[@"pid"];
        if (pidNum && [pidNum intValue] > 0) {
            kill((pid_t)[pidNum intValue], SIGTERM);
        }
        [[NSFileManager defaultManager] removeItemAtPath:statusFile error:nil];
    }
    NSString *sockFile = [@"~/.cache/decision-cli/worker.sock" stringByExpandingTildeInPath];
    [[NSFileManager defaultManager] removeItemAtPath:sockFile error:nil];
    [NSApp terminate:nil];
}

@end

static StatusBarApp *g_app = nil;

void run_status_bar_app() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        g_app = [[StatusBarApp alloc] init];
        [NSApp setDelegate:g_app];
        [NSApp run];
    }
}
*/
import "C"

import (
	"os"
	"path/filepath"
	"runtime"
	"syscall"

	"github.com/cafe3310/decision-cli/internal/config"
)

func init() {
	runtime.LockOSThread()
}

// AcquireBarLock attempts to acquire a non-blocking exclusive lock on ~/.cache/decision-cli/bar.lock.
// Returns the locked file if acquired, or nil if another status bar instance is already running.
func AcquireBarLock() *os.File {
	lockPath := config.ExpandPath("~/.cache/decision-cli/bar.lock")
	_ = os.MkdirAll(filepath.Dir(lockPath), 0755)
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = f.Close()
		return nil
	}
	return f
}

// IsBarRunning checks whether ~/.cache/decision-cli/bar.lock is currently held by an active bar process.
func IsBarRunning() bool {
	lockPath := config.ExpandPath("~/.cache/decision-cli/bar.lock")
	f, err := os.OpenFile(lockPath, os.O_RDONLY, 0644)
	if err != nil {
		return false
	}
	defer f.Close()
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		// Lock is held by running bar instance
		return true
	}
	_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	return false
}

// Run starts the native macOS status bar item on the main thread and polls worker status.
func Run() error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	lockFile := AcquireBarLock()
	if lockFile == nil {
		// Another status bar instance is already running
		return nil
	}
	defer func() {
		_ = syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)
		_ = lockFile.Close()
	}()

	C.run_status_bar_app()
	return nil
}
